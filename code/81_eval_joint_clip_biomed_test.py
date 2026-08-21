#!/usr/bin/env python3
"""One-shot canonical test evaluation for the frozen joint CLIP/BiomedCLIP scorer.

Safeguards
----------
* Requires the holdout-approved manifest produced by script 80.
* Loads the already-frozen three-seed ensemble; performs no training or tuning.
* Reproduces both canonical frozen baselines before computing the ensemble:
  BiomedCLIP 743/2000 and OpenAI CLIP 622/2000 by default.
* Refuses to overwrite an existing completed test result.
* Reports the result once, whether positive, null, or negative.
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
    p.add_argument("--test-csv", type=Path, required=True)
    p.add_argument("--image-root", type=Path, default=None)
    p.add_argument("--approved-manifest", type=Path, required=True)
    p.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    p.add_argument("--context-length", type=int, default=256)
    p.add_argument("--eval-limit", type=int, default=2000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--progress-every", type=int, default=100)
    p.add_argument("--bootstrap-repeats", type=int, default=50000)
    p.add_argument("--bootstrap-seed", type=int, default=42)
    p.add_argument("--expected-biomed-correct", type=int, default=743)
    p.add_argument("--expected-clip-correct", type=int, default=622)
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


def comparison_metrics(
    reference_correct_t: torch.Tensor,
    candidate_correct_t: torch.Tensor,
) -> Dict[str, Any]:
    n = int(reference_correct_t.numel())
    reference_correct = int(reference_correct_t.sum().item())
    candidate_correct = int(candidate_correct_t.sum().item())
    fixed = int((~reference_correct_t & candidate_correct_t).sum().item())
    broken = int((reference_correct_t & ~candidate_correct_t).sum().item())
    both_correct = int((reference_correct_t & candidate_correct_t).sum().item())
    both_wrong = int((~reference_correct_t & ~candidate_correct_t).sum().item())
    return {
        "reference_correct": reference_correct,
        "reference_accuracy": reference_correct / n,
        "candidate_correct": candidate_correct,
        "candidate_accuracy": candidate_correct / n,
        "gain_correct": candidate_correct - reference_correct,
        "accuracy_delta": (candidate_correct - reference_correct) / n,
        "both_correct": both_correct,
        "both_wrong": both_wrong,
        "fixed": fixed,
        "broken": broken,
        "mcnemar_exact_p": exact_mcnemar_p(fixed, broken),
    }


def main() -> int:
    args = parse_args()
    args.project_root = args.project_root.resolve()
    args.data_root = args.data_root.resolve() if args.data_root else None
    args.test_csv = args.test_csv.resolve()
    args.image_root = args.image_root.resolve() if args.image_root else None
    args.approved_manifest = args.approved_manifest.resolve()
    args.out_dir = args.out_dir.resolve()

    if args.eval_limit <= 0:
        raise ValueError("--eval-limit must be positive.")
    if not args.test_csv.is_file():
        raise FileNotFoundError(args.test_csv)
    if not args.approved_manifest.is_file():
        raise FileNotFoundError(args.approved_manifest)

    completed_summary = args.out_dir / "test_summary.json"
    if completed_summary.exists() and not args.allow_overwrite:
        raise RuntimeError(
            f"Completed canonical test result already exists: {completed_summary}. "
            "Refusing to overwrite the one-shot evaluation."
        )
    args.out_dir.mkdir(parents=True, exist_ok=True)

    train_mod = load_training_module()
    manifest = train_mod.safe_torch_load(args.approved_manifest)

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
        "holdout_gate_pass",
        "holdout_rows",
        "holdout_baseline_accuracy",
        "holdout_ensemble_accuracy",
        "holdout_ensemble_delta",
        "holdout_gain_correct",
        "holdout_required_gain_correct",
        "holdout_permutation_audit",
    }
    missing = sorted(required_fields.difference(manifest))
    if missing:
        raise RuntimeError(f"Approved manifest is missing fields: {missing}")
    if not bool(manifest["validation_gate_pass"]):
        raise RuntimeError("Development gate did not pass.")
    if not bool(manifest["holdout_gate_pass"]):
        raise RuntimeError("Holdout gate did not pass; test evaluation is refused.")
    if int(manifest["holdout_gain_correct"]) < int(
        manifest["holdout_required_gain_correct"]
    ):
        raise RuntimeError("Approved manifest contains an inconsistent holdout gate.")
    if not bool(manifest["holdout_permutation_audit"].get("passed", False)):
        raise RuntimeError("Holdout permutation audit did not pass.")
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
        args.test_csv,
        args.eval_limit,
        image_roots,
        image_index,
    )
    if len(parsed_rows) != args.eval_limit:
        raise RuntimeError(
            f"Expected {args.eval_limit} test rows, parsed {len(parsed_rows)}."
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
    test_biomed = train_mod.extract_biomed_features(
        split_name="test",
        parsed=parsed_rows,
        csv_path=args.test_csv,
        limit=args.eval_limit,
        cache_path=args.out_dir / "test_biomed_features.pt",
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

    test_clip = train_mod.extract_clip_features(
        split_name="test",
        parsed=parsed_rows,
        csv_path=args.test_csv,
        limit=args.eval_limit,
        cache_path=args.out_dir / "test_clip_features.pt",
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
    train_mod.assert_aligned(test_biomed, test_clip, "test")

    gold = test_biomed["gold_indices"].long()
    biomed_scores = test_biomed["base_scores"].float()
    clip_scores = test_clip["base_scores"].float()
    biomed_pred = biomed_scores.argmax(dim=1)
    clip_pred = clip_scores.argmax(dim=1)
    biomed_correct_t = biomed_pred.eq(gold)
    clip_correct_t = clip_pred.eq(gold)

    n = int(gold.numel())
    biomed_correct = int(biomed_correct_t.sum().item())
    clip_correct = int(clip_correct_t.sum().item())

    # Protocol reproduction is checked before the learned ensemble is computed.
    if biomed_correct != args.expected_biomed_correct:
        raise RuntimeError(
            "Canonical BiomedCLIP reproduction failed: "
            f"observed={biomed_correct}/{n}, "
            f"expected={args.expected_biomed_correct}/{n}. "
            "Do not compute or interpret the ensemble result."
        )
    if clip_correct != args.expected_clip_correct:
        raise RuntimeError(
            "Canonical CLIP reproduction failed: "
            f"observed={clip_correct}/{n}, "
            f"expected={args.expected_clip_correct}/{n}. "
            "Do not compute or interpret the ensemble result."
        )

    ensemble_logits = train_mod.ensemble_logits(
        state_paths,
        biomed=test_biomed,
        clip=test_clip,
        device=device,
        batch_size=args.batch_size,
    )
    ensemble_pred = ensemble_logits.argmax(dim=1)
    ensemble_correct_t = ensemble_pred.eq(gold)

    primary = comparison_metrics(biomed_correct_t, ensemble_correct_t)
    versus_clip = comparison_metrics(clip_correct_t, ensemble_correct_t)
    ensemble_correct = int(ensemble_correct_t.sum().item())
    ensemble_accuracy = ensemble_correct / n

    ci_low, ci_high = paired_bootstrap_ci(
        biomed_correct_t.numpy().astype(np.float64),
        ensemble_correct_t.numpy().astype(np.float64),
        repeats=args.bootstrap_repeats,
        seed=args.bootstrap_seed,
    )
    primary["paired_bootstrap_ci95"] = [ci_low, ci_high]

    permutation_audit = train_mod.permutation_equivariance_audit(
        state_paths,
        biomed=test_biomed,
        clip=test_clip,
        device=device,
        batch_size=args.batch_size,
    )
    if not permutation_audit["passed"]:
        raise RuntimeError(
            "Canonical test permutation-equivariance audit failed."
        )

    prediction_rows: List[Dict[str, Any]] = []
    for index, row in enumerate(test_biomed["rows"]):
        biomed_index = int(biomed_pred[index].item())
        clip_index = int(clip_pred[index].item())
        ensemble_index = int(ensemble_pred[index].item())
        if not biomed_correct_t[index] and ensemble_correct_t[index]:
            transition = "fixed"
        elif biomed_correct_t[index] and not ensemble_correct_t[index]:
            transition = "broken"
        elif biomed_correct_t[index] and ensemble_correct_t[index]:
            transition = "both_correct"
        else:
            transition = "both_wrong"
        prediction_rows.append(
            {
                **row,
                "biomedclip_pred_label": chr(ord("A") + biomed_index),
                "biomedclip_correct": int(biomed_correct_t[index].item()),
                "clip_pred_label": chr(ord("A") + clip_index),
                "clip_correct": int(clip_correct_t[index].item()),
                "ensemble_pred_label": chr(ord("A") + ensemble_index),
                "ensemble_correct": int(ensemble_correct_t[index].item()),
                "transition_vs_biomedclip": transition,
                "biomedclip_scores_json": json.dumps(
                    biomed_scores[index].tolist()
                ),
                "clip_scores_json": json.dumps(
                    clip_scores[index].tolist()
                ),
                "ensemble_logits_json": json.dumps(
                    ensemble_logits[index].tolist()
                ),
            }
        )
    write_csv(args.out_dir / "test_predictions.csv", prediction_rows)

    summary: Dict[str, Any] = {
        "experiment": "joint_clip_biomed_semantic_residual_canonical_test_v1",
        "protocol": "one_shot_holdout_approved_three_seed_ensemble",
        "status": "exploratory_after_prior_validation_experiments",
        "test_csv": str(args.test_csv),
        "test_csv_sha256": sha256_file(args.test_csv),
        "approved_manifest": str(args.approved_manifest),
        "approved_manifest_sha256": sha256_file(args.approved_manifest),
        "evaluated_rows": n,
        "biomed_model_id": train_mod.BIOMED_MODEL_ID,
        "clip_model_id": train_mod.CLIP_MODEL_ID,
        "prompt_template": train_mod.PROMPT_TEMPLATE,
        "development_and_holdout_selection": {
            "seeds": manifest["seeds"],
            "validation_gate_pass": bool(manifest["validation_gate_pass"]),
            "validation_baseline_accuracy": float(
                manifest["validation_baseline_accuracy"]
            ),
            "validation_ensemble_accuracy": float(
                manifest["validation_ensemble_accuracy"]
            ),
            "validation_ensemble_delta": float(
                manifest["validation_ensemble_delta"]
            ),
            "holdout_gate_pass": bool(manifest["holdout_gate_pass"]),
            "holdout_rows": int(manifest["holdout_rows"]),
            "holdout_baseline_accuracy": float(
                manifest["holdout_baseline_accuracy"]
            ),
            "holdout_ensemble_accuracy": float(
                manifest["holdout_ensemble_accuracy"]
            ),
            "holdout_ensemble_delta": float(
                manifest["holdout_ensemble_delta"]
            ),
            "holdout_gain_correct": int(manifest["holdout_gain_correct"]),
            "holdout_required_gain_correct": int(
                manifest["holdout_required_gain_correct"]
            ),
        },
        "canonical_baseline_reproduction": {
            "biomedclip": {
                "correct": biomed_correct,
                "accuracy": biomed_correct / n,
                "expected_correct": args.expected_biomed_correct,
                "reproduced": True,
                "prediction_distribution": label_distribution(biomed_pred),
            },
            "clip": {
                "correct": clip_correct,
                "accuracy": clip_correct / n,
                "expected_correct": args.expected_clip_correct,
                "reproduced": True,
                "prediction_distribution": label_distribution(clip_pred),
            },
        },
        "joint_ensemble": {
            "correct": ensemble_correct,
            "accuracy": ensemble_accuracy,
            "prediction_distribution": label_distribution(ensemble_pred),
        },
        "primary_comparison_vs_biomedclip": primary,
        "secondary_comparison_vs_clip": versus_clip,
        "permutation_equivariance_audit": permutation_audit,
        "runtime": {
            "feature_extraction_seconds": feature_extraction_seconds,
            "bootstrap_repeats": args.bootstrap_repeats,
            "bootstrap_seed": args.bootstrap_seed,
        },
        "interpretation_rule": (
            "Report this canonical test result once, whether positive, null, or "
            "negative. Do not retune the scorer, select a seed, alter the data "
            "split, or choose another architecture after observing it."
        ),
    }
    write_json(args.out_dir / "test_summary.json", summary)

    train_mod.log(
        f"canonical_biomedclip={biomed_correct}/{n} "
        f"({biomed_correct/n:.4f})"
    )
    train_mod.log(
        f"canonical_clip={clip_correct}/{n} ({clip_correct/n:.4f})"
    )
    train_mod.log(
        f"canonical_ensemble={ensemble_correct}/{n} "
        f"({ensemble_accuracy:.4f}) "
        f"delta_vs_biomedclip={primary['accuracy_delta']:+.4f}"
    )
    train_mod.log(
        f"fixed={primary['fixed']} broken={primary['broken']} "
        f"mcnemar_exact_p={primary['mcnemar_exact_p']:.6g} "
        f"paired_ci95=[{ci_low:.4f},{ci_high:.4f}]"
    )
    train_mod.log(f"summary={args.out_dir / 'test_summary.json'}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
