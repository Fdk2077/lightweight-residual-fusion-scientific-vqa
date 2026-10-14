#!/usr/bin/env python3
"""End-to-end efficiency audit for the frozen CLIP/BiomedCLIP residual VQA pipeline.

Purpose
-------
Measure computational cost without changing model selection or test predictions.
The audit is intended for a fixed evaluation split (preferably the prespecified
validation/development split) and reports:

* encoder parameter counts;
* residual-head parameters per seed and across the three-seed ensemble;
* model-load time;
* feature-extraction/inference time;
* scorer-only time;
* peak CUDA allocation under the sequential low-VRAM implementation;
* throughput on the audited rows.

The script performs no training and no hyperparameter selection.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import torch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--repo-code-dir", type=Path, required=True,
                   help="Directory containing 79_train_joint_clip_biomed_residual.py")
    p.add_argument("--project-root", type=Path, default=Path("."))
    p.add_argument("--data-root", type=Path, default=None)
    p.add_argument("--eval-csv", type=Path, required=True,
                   help="Fixed split used only for timing/efficiency; dev/validation is recommended.")
    p.add_argument("--image-root", type=Path, default=None)
    p.add_argument("--approved-manifest", type=Path, required=True,
                   help="Frozen manifest containing the three residual scorer state_paths.")
    p.add_argument("--eval-limit", type=int, default=256)
    p.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    p.add_argument("--context-length", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--progress-every", type=int, default=64)
    p.add_argument("--local-files-only", action="store_true")
    p.add_argument("--out-dir", type=Path, required=True)
    return p.parse_args()


def load_train_module(code_dir: Path) -> Any:
    code_dir = code_dir.resolve()
    direct = [
        code_dir / "79_train_joint_clip_biomed_residual.py",
        code_dir / "79_train_joint_clip_biomed_residual_v1.py",
    ]
    candidates = [p for p in direct if p.is_file()]
    if not candidates:
        for name in ("79_train_joint_clip_biomed_residual.py", "79_train_joint_clip_biomed_residual_v1.py"):
            candidates.extend(code_dir.rglob(name))
    unique = []
    seen = set()
    for p in candidates:
        rp = p.resolve()
        if str(rp).lower() not in seen:
            seen.add(str(rp).lower()); unique.append(rp)
    if len(unique) != 1:
        raise RuntimeError(
            f"Expected exactly one residual training module under {code_dir}; found: "
            + ", ".join(map(str, unique))
        )
    path = unique[0]
    spec = importlib.util.spec_from_file_location("joint_residual_current", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def reset_peak(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def mem_gb(device: torch.device) -> Dict[str, float]:
    if device.type != "cuda":
        return {"gpu_current_allocated_gb": 0.0, "gpu_peak_allocated_gb": 0.0}
    return {
        "gpu_current_allocated_gb": torch.cuda.memory_allocated(device) / (1024 ** 3),
        "gpu_peak_allocated_gb": torch.cuda.max_memory_allocated(device) / (1024 ** 3),
    }


def count_params(model: torch.nn.Module) -> Dict[str, int]:
    return {
        "total_parameters": int(sum(p.numel() for p in model.parameters())),
        "trainable_parameters": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
    }


def accuracy(scores: torch.Tensor, gold: torch.Tensor) -> float:
    return float(scores.argmax(dim=1).eq(gold).float().mean().item())


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
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
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main() -> int:
    args = parse_args()
    args.repo_code_dir = args.repo_code_dir.resolve()
    args.project_root = args.project_root.resolve()
    args.data_root = args.data_root.resolve() if args.data_root else None
    args.eval_csv = args.eval_csv.resolve()
    args.image_root = args.image_root.resolve() if args.image_root else None
    args.approved_manifest = args.approved_manifest.resolve()
    args.out_dir = args.out_dir.resolve()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.eval_limit <= 0:
        raise ValueError("--eval-limit must be positive")
    if not args.eval_csv.is_file():
        raise FileNotFoundError(args.eval_csv)
    if not args.approved_manifest.is_file():
        raise FileNotFoundError(args.approved_manifest)

    tm = load_train_module(args.repo_code_dir)
    device = tm.choose_device(args.device)
    tm.log(f"efficiency_device={device}")
    if device.type == "cuda":
        tm.log(f"cuda_device={torch.cuda.get_device_name(device)}")

    image_roots = ([args.image_root] if args.image_root else
                   tm.discover_image_roots(args.project_root, args.data_root))
    image_roots = [p.resolve() for p in image_roots if p and p.is_dir()]
    if not image_roots:
        raise RuntimeError("No image root found. Pass --image-root.")
    image_index = tm.index_images(image_roots)
    parsed = tm.parse_rows(args.eval_csv, args.eval_limit, image_roots, image_index)
    n = len(parsed)

    manifest = tm.safe_torch_load(args.approved_manifest)
    state_paths = [Path(p) for p in manifest.get("state_paths", [])]
    if len(state_paths) != 3:
        raise RuntimeError(f"Expected three frozen seed states, got {len(state_paths)}")
    missing = [str(p) for p in state_paths if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"Frozen state files not found: {missing}")

    rows: List[Dict[str, Any]] = []

    # ---- BiomedCLIP ----
    from open_clip import create_model_from_pretrained, get_tokenizer
    sync(device)
    t0 = time.perf_counter()
    biomed_model, biomed_preprocess = create_model_from_pretrained(tm.BIOMED_MODEL_ID)
    biomed_tokenizer = get_tokenizer(tm.BIOMED_MODEL_ID)
    biomed_model = biomed_model.to(device).eval()
    for p in biomed_model.parameters():
        p.requires_grad_(False)
    sync(device)
    biomed_load_s = time.perf_counter() - t0
    biomed_params = count_params(biomed_model)
    reset_peak(device)
    t0 = time.perf_counter()
    biomed = tm.extract_biomed_features(
        split_name="efficiency",
        parsed=parsed,
        csv_path=args.eval_csv,
        limit=args.eval_limit,
        cache_path=args.out_dir / "efficiency_biomed_features.pt",
        rebuild_cache=True,
        model=biomed_model,
        preprocess=biomed_preprocess,
        tokenizer=biomed_tokenizer,
        device=device,
        context_length=args.context_length,
        progress_every=args.progress_every,
    )
    sync(device)
    biomed_infer_s = time.perf_counter() - t0
    biomed_mem = mem_gb(device)
    biomed_acc = accuracy(biomed["base_scores"].float(), biomed["gold_indices"].long())
    rows.append({
        "system": "BiomedCLIP",
        "rows": n,
        **biomed_params,
        "stored_trainable_parameters": 0,
        "load_seconds": biomed_load_s,
        "inference_seconds": biomed_infer_s,
        "total_seconds_excluding_download": biomed_load_s + biomed_infer_s,
        "rows_per_second_inference": n / biomed_infer_s,
        "accuracy_on_audit_split": biomed_acc,
        **biomed_mem,
        "implementation": "single frozen encoder",
    })
    del biomed_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # ---- OpenAI CLIP ----
    from transformers import AutoImageProcessor, AutoTokenizer, CLIPModel
    sync(device)
    t0 = time.perf_counter()
    clip_model = CLIPModel.from_pretrained(
        tm.CLIP_MODEL_ID, local_files_only=args.local_files_only
    )
    clip_tokenizer = AutoTokenizer.from_pretrained(
        tm.CLIP_MODEL_ID, local_files_only=args.local_files_only
    )
    clip_image_processor = AutoImageProcessor.from_pretrained(
        tm.CLIP_MODEL_ID, local_files_only=args.local_files_only
    )
    clip_model = clip_model.to(device).eval()
    for p in clip_model.parameters():
        p.requires_grad_(False)
    clip_context_length = int(getattr(clip_model.config.text_config, "max_position_embeddings", 77))
    sync(device)
    clip_load_s = time.perf_counter() - t0
    clip_params = count_params(clip_model)
    reset_peak(device)
    t0 = time.perf_counter()
    clip = tm.extract_clip_features(
        split_name="efficiency",
        parsed=parsed,
        csv_path=args.eval_csv,
        limit=args.eval_limit,
        cache_path=args.out_dir / "efficiency_clip_features.pt",
        rebuild_cache=True,
        model=clip_model,
        image_processor=clip_image_processor,
        tokenizer=clip_tokenizer,
        device=device,
        context_length=clip_context_length,
        progress_every=args.progress_every,
    )
    sync(device)
    clip_infer_s = time.perf_counter() - t0
    clip_mem = mem_gb(device)
    clip_acc = accuracy(clip["base_scores"].float(), clip["gold_indices"].long())
    rows.append({
        "system": "OpenAI CLIP",
        "rows": n,
        **clip_params,
        "stored_trainable_parameters": 0,
        "load_seconds": clip_load_s,
        "inference_seconds": clip_infer_s,
        "total_seconds_excluding_download": clip_load_s + clip_infer_s,
        "rows_per_second_inference": n / clip_infer_s,
        "accuracy_on_audit_split": clip_acc,
        **clip_mem,
        "implementation": "single frozen encoder",
    })
    del clip_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    tm.assert_aligned(biomed, clip, "efficiency")

    # ---- Three-seed residual ensemble head on already-extracted features ----
    first_head, _ = tm.load_seed_model(state_paths[0], device)
    per_seed_head_params = count_params(first_head)
    del first_head
    if device.type == "cuda":
        torch.cuda.empty_cache()

    reset_peak(device)
    sync(device)
    t0 = time.perf_counter()
    logits = tm.ensemble_logits(
        state_paths,
        biomed=biomed,
        clip=clip,
        device=device,
        batch_size=args.batch_size,
    )
    sync(device)
    head_s = time.perf_counter() - t0
    head_mem = mem_gb(device)
    full_acc = accuracy(logits.float(), biomed["gold_indices"].long())

    joint_total_params = (
        biomed_params["total_parameters"]
        + clip_params["total_parameters"]
        + per_seed_head_params["total_parameters"] * len(state_paths)
    )
    joint_stored_trainable = per_seed_head_params["trainable_parameters"] * len(state_paths)
    sequential_peak = max(
        biomed_mem["gpu_peak_allocated_gb"],
        clip_mem["gpu_peak_allocated_gb"],
        head_mem["gpu_peak_allocated_gb"],
    )
    joint_infer_s = biomed_infer_s + clip_infer_s + head_s
    joint_load_s = biomed_load_s + clip_load_s
    rows.append({
        "system": "Full residual ensemble (3 seeds)",
        "rows": n,
        "total_parameters": joint_total_params,
        "trainable_parameters": per_seed_head_params["trainable_parameters"],
        "stored_trainable_parameters": joint_stored_trainable,
        "load_seconds": joint_load_s,
        "inference_seconds": joint_infer_s,
        "total_seconds_excluding_download": joint_load_s + joint_infer_s,
        "rows_per_second_inference": n / joint_infer_s,
        "accuracy_on_audit_split": full_acc,
        "gpu_current_allocated_gb": head_mem["gpu_current_allocated_gb"],
        "gpu_peak_allocated_gb": sequential_peak,
        "head_only_seconds": head_s,
        "head_only_peak_gpu_gb": head_mem["gpu_peak_allocated_gb"],
        "head_parameters_per_seed": per_seed_head_params["total_parameters"],
        "ensemble_heads": len(state_paths),
        "implementation": "sequential encoders + three frozen residual heads",
    })

    summary = {
        "analysis": "efficiency_audit_v1",
        "status": "posthoc_descriptive_no_model_selection",
        "eval_csv": str(args.eval_csv),
        "rows": n,
        "device": str(device),
        "cuda_device": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "notes": [
            "Times exclude model download/network transfer.",
            "Encoder inference is measured sequentially to match the low-VRAM implementation.",
            "The manuscript's 32,961 trainable-parameter figure is a per-seed head count; the deployed three-seed ensemble stores three such heads.",
            "This audit changes no model weights, prompts, thresholds, or selection criteria.",
        ],
        "systems": rows,
    }
    (args.out_dir / "efficiency_audit_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    write_csv(args.out_dir / "efficiency_audit.csv", rows)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())