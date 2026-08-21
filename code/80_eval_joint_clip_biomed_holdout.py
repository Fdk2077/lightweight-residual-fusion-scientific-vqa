#!/usr/bin/env python3
"""One-shot group-disjoint holdout evaluation for the frozen joint scorer.

This script:
* loads the already-frozen three-seed ensemble from script 79;
* extracts canonical BiomedCLIP and OpenAI CLIP features for the holdout only;
* evaluates the ensemble without training or parameter selection;
* compares it pairwise with frozen BiomedCLIP;
* audits option-permutation equivariance;
* approves canonical-test evaluation only when the predeclared holdout gate passes.

The holdout gate defaults to an absolute gain of at least 0.02. On 290 rows,
this means at least six additional correct answers.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
TRAIN_SCRIPT = SCRIPT_DIR / "79_train_joint_clip_biomed_residual_v1.py"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--project-root", type=Path, default=Path("."))
    p.add_argument("--data-root", type=Path, default=None)
    p.add_argument("--holdout-csv", type=Path, required=True)
    p.add_argument("--image-root", type=Path, default=None)
    p.add_argument("--ensemble-manifest", type=Path, required=True)
    p.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    p.add_argument("--context-length", type=int, default=256)
    p.add_argument("--eval-limit", type=int, default=290)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--progress-every", type=int, default=50)
    p.add_argument("--bootstrap-repeats", type=int, default=50000)
    p.add_argument("--bootstrap-seed", type=int, default=42)
    p.add_argument(
        "--min-holdout-delta",
        type=float,
        default=0.02,
        help="Absolute accuracy gain over frozen BiomedCLIP required for test approval.",
    )
    p.add_argument("--local-files-only", action="store_true")
    p.add_argument("--rebuild-cache", action="store_true")
    p.add_argument("--allow-overwrite", action="store_true")
    p.add_argument("--out-dir", type=Path, required=True)
    return p.parse_args()


def load_training_module() -> Any:
    if not TRAIN_SCRIPT.is_file():
        raise FileNotFoundError(
            f"Required sibling training script not found: {TRAIN_SCRIPT}"
        )
    spec = importlib.util.spec_from_file_location(
        "joint_clip_biomed_train_v1",
        TRAIN_SCRIPT,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {TRAIN_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields: List[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def exact_mcnemar_p(fixed: int, broken: int) -> float:
    n = fixed + broken
    if n == 0:
        return 1.0
    k = min(fixed, broken)
    lower_tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2.0 * lower_tail)


def paired_bootstrap_ci(
    baseline_correct: np.ndarray,
    ensemble_correct: np.ndarray,
    repeats: int,
    seed: int,
) -> Tuple[float, float]:
    if repeats <= 0:
        raise ValueError("--bootstrap-repeats must be positive.")
    if baseline_correct.shape != ensemble_correct.shape:
        raise ValueError("Bootstrap arrays are not aligned.")

    rng = np.random.default_rng(seed)
    n = baseline_correct.size
    deltas = np.empty(repeats, dtype=np.float64)
    chunk_size = 1000
    done = 0
    while done < repeats:
        size = min(chunk_size, repeats - done)
        indices = rng.integers(0, n, size=(size, n), endpoint=False)
        deltas[done : done + size] = (
            ensemble_correct[indices].mean(axis=1)
            - baseline_correct[indices].mean(axis=1)
        )
        done += size
    low, high = np.quantile(deltas, [0.025, 0.975])
    return float(low), float(high)


def label_distribution(predictions: torch.Tensor) -> Dict[str, Any]:
    counts = Counter(
        chr(ord("A") + int(index))
        for index in predictions.tolist()
    )
    n = int(predictions.numel())
    return {
        "counts": {label: int(counts.get(label, 0)) for label in "ABCD"},
        "rates": {
            label: counts.get(label, 0) / n
            for label in "ABCD"
        },
        "max_rate": max(counts.get(label, 0) / n for label in "ABCD"),
        "min_rate": min(counts.get(label, 0) / n for label in "ABCD"),
    }


def main() -> int:
    args = parse_args()
    args.project_root = args.project_root.resolve()
    args.data_root = args.data_root.resolve() if args.data_root else None
    args.holdout_csv = args.holdout_csv.resolve()
    args.image_root = args.image_root.resolve() if args.image_root else None
    args.ensemble_manifest = args.ensemble_manifest.resolve()
    args.out_dir = args.out_dir.resolve()

    if args.eval_limit <= 0:
        raise ValueError("--eval-limit must be positive.")
    if args.min_holdout_delta <= 0:
        raise ValueError("--min-holdout-delta must be positive.")
    if not args.holdout_csv.is_file():
        raise FileNotFoundError(args.holdout_csv)
    if not args.ensemble_manifest.is_file():
        raise FileNotFoundError(args.ensemble_manifest)

    completed_summary = args.out_dir / "holdout_summary.json"
    if completed_summary.exists() and not args.allow_overwrite:
        raise RuntimeError(
            f"Completed holdout result already exists: {completed_summary}. "
            "Refusing to overwrite the one-shot evaluation."
        )
    args.out_dir.mkdir(parents=True, exist_ok=True)

    train_mod = load_training_module()
    manifest = train_mod.safe_torch_load(args.ensemble_manifest)

    required_fields = {
        "experiment",
        "biomed_model_id",
        "clip_model_id",
        "prompt_template",
        "biomed_score_protocol",
        "clip_score_protocol",
        "state_paths",
        "seeds",
        "validation_gate_pass",
        "validation_baseline_accuracy",
        "validation_ensemble_accuracy",
        "validation_ensemble_delta",
    }
    missing = sorted(required_fields.difference(manifest))
    if missing:
        raise RuntimeError(f"Manifest is missing fields: {missing}")
    if not bool(manifest["validation_gate_pass"]):
        raise RuntimeError(
            "Development gate did not pass; holdout evaluation is not permitted."
        )
    if manifest["biomed_model_id"] != train_mod.BIOMED_MODEL_ID:
        raise RuntimeError("BiomedCLIP model mismatch.")
    if manifest["clip_model_id"] != train_mod.CLIP_MODEL_ID:
        raise RuntimeError("CLIP model mismatch.")
    if manifest["prompt_template"] != train_mod.PROMPT_TEMPLATE:
        raise RuntimeError("Prompt-template mismatch.")
    if manifest["biomed_score_protocol"] != train_mod.BIOMED_SCORE_PROTOCOL:
        raise RuntimeError("BiomedCLIP score-protocol mismatch.")
    if manifest["clip_score_protocol"] != train_mod.CLIP_SCORE_PROTOCOL:
        raise RuntimeError("CLIP score-protocol mismatch.")

    state_paths = [Path(path) for path in manifest["state_paths"]]
    if len(state_paths) != 3:
        raise RuntimeError(
            f"Expected exactly three frozen seed states, found {len(state_paths)}."
        )
    missing_states = [str(path) for path in state_paths if not path.is_file()]
    if missing_states:
        raise FileNotFoundError(
            f"Frozen seed state files are missing: {missing_states}"
        )

    device = train_mod.choose_device(args.device)
    train_mod.log(f"device={device}")
    if device.type == "cuda":
        train_mod.log(f"cuda_device={torch.cuda.get_device_name(0)}")

    image_roots = (
        [args.image_root]
        if args.image_root
        else train_mod.discover_image_roots(args.project_root, args.data_root)
    )
    image_roots = [
        path.resolve()
        for path in image_roots
        if path and path.is_dir()
    ]
    if not image_roots:
        raise RuntimeError("No image root found. Pass --image-root.")
    image_index = train_mod.index_images(image_roots)
    train_mod.log(f"indexed_image_keys={len(image_index)}")

    parsed_rows = train_mod.parse_rows(
        args.holdout_csv,
        args.eval_limit,
        image_roots,
        image_index,
    )
    if len(parsed_rows) != args.eval_limit:
        raise RuntimeError(
            f"Expected {args.eval_limit} holdout rows, parsed {len(parsed_rows)}."
        )

    try:
        from open_clip import create_model_from_pretrained, get_tokenizer
    except ImportError as exc:
        raise RuntimeError("open_clip is not installed.") from exc

    train_mod.log(f"loading_biomed_model={train_mod.BIOMED_MODEL_ID}")
    biomed_model, biomed_preprocess = create_model_from_pretrained(
        train_mod.BIOMED_MODEL_ID
    )
    biomed_tokenizer = get_tokenizer(train_mod.BIOMED_MODEL_ID)
    biomed_model = biomed_model.to(device)
    biomed_model.eval()
    for parameter in biomed_model.parameters():
        parameter.requires_grad_(False)

    started = time.time()
    holdout_biomed = train_mod.extract_biomed_features(
        split_name="holdout",
        parsed=parsed_rows,
        csv_path=args.holdout_csv,
        limit=args.eval_limit,
        cache_path=args.out_dir / "holdout_biomed_features.pt",
        rebuild_cache=args.rebuild_cache,
        model=biomed_model,
        preprocess=biomed_preprocess,
        tokenizer=biomed_tokenizer,
        device=device,
        context_length=args.context_length,
        progress_every=args.progress_every,
    )
    del biomed_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    try:
        from transformers import AutoImageProcessor, AutoTokenizer, CLIPModel
    except ImportError as exc:
        raise RuntimeError("transformers is not installed.") from exc

    train_mod.log(f"loading_clip_model={train_mod.CLIP_MODEL_ID}")
    clip_model = CLIPModel.from_pretrained(
        train_mod.CLIP_MODEL_ID,
        local_files_only=args.local_files_only,
    )
    clip_tokenizer = AutoTokenizer.from_pretrained(
        train_mod.CLIP_MODEL_ID,
        local_files_only=args.local_files_only,
    )
    clip_image_processor = AutoImageProcessor.from_pretrained(
        train_mod.CLIP_MODEL_ID,
        local_files_only=args.local_files_only,
    )
    clip_model = clip_model.to(device)
    clip_model.eval()
    for parameter in clip_model.parameters():
        parameter.requires_grad_(False)
    clip_context_length = int(
        getattr(
            clip_model.config.text_config,
            "max_position_embeddings",
            77,
        )
    )

    holdout_clip = train_mod.extract_clip_features(
        split_name="holdout",
        parsed=parsed_rows,
        csv_path=args.holdout_csv,
        limit=args.eval_limit,
        cache_path=args.out_dir / "holdout_clip_features.pt",
        rebuild_cache=args.rebuild_cache,
        model=clip_model,
        image_processor=clip_image_processor,
        tokenizer=clip_tokenizer,
        device=device,
        context_length=clip_context_length,
        progress_every=args.progress_every,
    )
    del clip_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    feature_extraction_seconds = time.time() - started
    train_mod.assert_aligned(holdout_biomed, holdout_clip, "holdout")

    gold = holdout_biomed["gold_indices"].long()
    baseline_scores = holdout_biomed["base_scores"].float()
    baseline_pred = baseline_scores.argmax(dim=1)
    baseline_correct_t = baseline_pred.eq(gold)

    ensemble_logits = train_mod.ensemble_logits(
        state_paths,
        biomed=holdout_biomed,
        clip=holdout_clip,
        device=device,
        batch_size=args.batch_size,
    )
    ensemble_pred = ensemble_logits.argmax(dim=1)
    ensemble_correct_t = ensemble_pred.eq(gold)

    n = int(gold.numel())
    baseline_correct = int(baseline_correct_t.sum().item())
    ensemble_correct = int(ensemble_correct_t.sum().item())
    baseline_accuracy = baseline_correct / n
    ensemble_accuracy = ensemble_correct / n
    gain_correct = ensemble_correct - baseline_correct
    delta = ensemble_accuracy - baseline_accuracy

    fixed = int((~baseline_correct_t & ensemble_correct_t).sum().item())
    broken = int((baseline_correct_t & ~ensemble_correct_t).sum().item())
    both_correct = int((baseline_correct_t & ensemble_correct_t).sum().item())
    both_wrong = int((~baseline_correct_t & ~ensemble_correct_t).sum().item())
    mcnemar_p = exact_mcnemar_p(fixed, broken)

    ci_low, ci_high = paired_bootstrap_ci(
        baseline_correct_t.numpy().astype(np.float64),
        ensemble_correct_t.numpy().astype(np.float64),
        repeats=args.bootstrap_repeats,
        seed=args.bootstrap_seed,
    )

    permutation_audit = train_mod.permutation_equivariance_audit(
        state_paths,
        biomed=holdout_biomed,
        clip=holdout_clip,
        device=device,
        batch_size=args.batch_size,
    )

    required_gain = max(
        1,
        math.ceil(args.min_holdout_delta * n - 1e-12),
    )
    gate_pass = (
        gain_correct >= required_gain
        and permutation_audit["passed"]
    )

    prediction_rows: List[Dict[str, Any]] = []
    for index, row in enumerate(holdout_biomed["rows"]):
        baseline_index = int(baseline_pred[index].item())
        ensemble_index = int(ensemble_pred[index].item())
        if not baseline_correct_t[index] and ensemble_correct_t[index]:
            transition = "fixed"
        elif baseline_correct_t[index] and not ensemble_correct_t[index]:
            transition = "broken"
        elif baseline_correct_t[index] and ensemble_correct_t[index]:
            transition = "both_correct"
        else:
            transition = "both_wrong"
        prediction_rows.append(
            {
                **row,
                "baseline_pred_label": chr(ord("A") + baseline_index),
                "baseline_correct": int(baseline_correct_t[index].item()),
                "ensemble_pred_label": chr(ord("A") + ensemble_index),
                "ensemble_correct": int(ensemble_correct_t[index].item()),
                "transition": transition,
                "baseline_scores_json": json.dumps(
                    baseline_scores[index].tolist()
                ),
                "ensemble_logits_json": json.dumps(
                    ensemble_logits[index].tolist()
                ),
            }
        )
    write_csv(args.out_dir / "holdout_predictions.csv", prediction_rows)

    summary: Dict[str, Any] = {
        "experiment": "joint_clip_biomed_semantic_residual_holdout_v1",
        "protocol": "one_shot_frozen_three_seed_ensemble",
        "status": "exploratory_after_prior_validation_experiments",
        "holdout_csv": str(args.holdout_csv),
        "holdout_csv_sha256": sha256_file(args.holdout_csv),
        "source_ensemble_manifest": str(args.ensemble_manifest),
        "source_ensemble_manifest_sha256": sha256_file(
            args.ensemble_manifest
        ),
        "evaluated_rows": n,
        "biomed_model_id": train_mod.BIOMED_MODEL_ID,
        "clip_model_id": train_mod.CLIP_MODEL_ID,
        "prompt_template": train_mod.PROMPT_TEMPLATE,
        "development_selection": {
            "seeds": manifest["seeds"],
            "validation_gate_pass": bool(
                manifest["validation_gate_pass"]
            ),
            "validation_baseline_accuracy": float(
                manifest["validation_baseline_accuracy"]
            ),
            "validation_ensemble_accuracy": float(
                manifest["validation_ensemble_accuracy"]
            ),
            "validation_ensemble_delta": float(
                manifest["validation_ensemble_delta"]
            ),
        },
        "frozen_biomedclip": {
            "correct": baseline_correct,
            "accuracy": baseline_accuracy,
            "prediction_distribution": label_distribution(
                baseline_pred
            ),
        },
        "joint_ensemble": {
            "correct": ensemble_correct,
            "accuracy": ensemble_accuracy,
            "gain_correct": gain_correct,
            "accuracy_delta": delta,
            "paired_bootstrap_ci95": [ci_low, ci_high],
            "prediction_distribution": label_distribution(
                ensemble_pred
            ),
        },
        "paired_transitions": {
            "both_correct": both_correct,
            "both_wrong": both_wrong,
            "fixed": fixed,
            "broken": broken,
            "mcnemar_exact_p": mcnemar_p,
        },
        "permutation_equivariance_audit": permutation_audit,
        "holdout_gate": {
            "min_accuracy_delta": args.min_holdout_delta,
            "required_gain_correct": required_gain,
            "observed_gain_correct": gain_correct,
            "passed": gate_pass,
        },
        "runtime": {
            "feature_extraction_seconds": feature_extraction_seconds,
            "bootstrap_repeats": args.bootstrap_repeats,
            "bootstrap_seed": args.bootstrap_seed,
        },
        "next_action": (
            "Evaluate the approved frozen ensemble once on the canonical "
            "2,000-row test set."
            if gate_pass
            else "Stop. Do not evaluate this scorer on the canonical test set."
        ),
    }
    write_json(args.out_dir / "holdout_summary.json", summary)

    approved_manifest_path = (
        args.out_dir / "ensemble_manifest_holdout_approved.pt"
    )
    if gate_pass:
        approved_manifest = dict(manifest)
        approved_manifest.update(
            {
                "holdout_gate_pass": True,
                "holdout_csv": str(args.holdout_csv),
                "holdout_csv_sha256": sha256_file(
                    args.holdout_csv
                ),
                "holdout_rows": n,
                "holdout_baseline_accuracy": baseline_accuracy,
                "holdout_ensemble_accuracy": ensemble_accuracy,
                "holdout_ensemble_delta": delta,
                "holdout_gain_correct": gain_correct,
                "holdout_required_gain_correct": required_gain,
                "holdout_permutation_audit": permutation_audit,
            }
        )
        torch.save(approved_manifest, approved_manifest_path)
        summary["approved_manifest"] = str(
            approved_manifest_path
        )
        write_json(args.out_dir / "holdout_summary.json", summary)

    train_mod.log(
        f"holdout_baseline={baseline_correct}/{n} "
        f"({baseline_accuracy:.4f})"
    )
    train_mod.log(
        f"holdout_ensemble={ensemble_correct}/{n} "
        f"({ensemble_accuracy:.4f}) delta={delta:+.4f}"
    )
    train_mod.log(
        f"fixed={fixed} broken={broken} "
        f"mcnemar_exact_p={mcnemar_p:.6g} "
        f"paired_ci95=[{ci_low:.4f},{ci_high:.4f}]"
    )
    train_mod.log(
        f"holdout_gate_passed={gate_pass} "
        f"required_gain={required_gain} observed_gain={gain_correct}"
    )
    train_mod.log(f"summary={args.out_dir / 'holdout_summary.json'}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
