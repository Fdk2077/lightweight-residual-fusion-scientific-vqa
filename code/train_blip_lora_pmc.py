from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image
from peft import LoraConfig, get_peft_model
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
from transformers import AutoProcessor, BlipForQuestionAnswering


MODEL_ID = "Salesforce/blip-vqa-base"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Memory-conscious LoRA smoke training for BLIP VQA on PMC-VQA."
        )
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path(
            "data/splits/pmc_vqa_v1/pmc_train_smoke_200.csv"
        ),
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=Path("data/processed/pmc_vqa_v1/images"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/lora/pmc_smoke_200_r4"),
    )
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--max-gradient-norm", type=float, default=1.0)
    parser.add_argument("--max-question-length", type=int, default=32)
    parser.add_argument("--max-answer-length", type=int, default=16)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--alpha", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument(
        "--last-decoder-layers",
        type=int,
        default=2,
        help="Apply LoRA only to the last N text-decoder layers.",
    )
    parser.add_argument(
        "--attention-scope",
        choices=("all", "cross"),
        default="all",
        help=(
            "'all' targets decoder self-attention and cross-attention "
            "query/value modules; 'cross' targets cross-attention only."
        ),
    )
    parser.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        help="Enable gradient checkpointing if additional memory saving is needed.",
    )
    parser.add_argument(
        "--grad-scaler-init-scale",
        type=float,
        default=1024.0,
        help="Initial AMP GradScaler scale. A conservative value reduces FP16 overflow.",
    )
    parser.add_argument(
        "--grad-scaler-growth-interval",
        type=int,
        default=1000,
        help="Successful steps before GradScaler increases its scale.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def image_basename(value: Any) -> str:
    return Path(str(value).replace("\\", "/")).name


def clean_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value).strip())


def discover_lora_targets(
    model: torch.nn.Module,
    last_decoder_layers: int,
    attention_scope: str,
) -> tuple[list[str], list[int]]:
    candidates: list[tuple[str, int]] = []

    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        if "text_decoder" not in name:
            continue
        if not (name.endswith(".query") or name.endswith(".value")):
            continue
        if attention_scope == "cross" and ".crossattention." not in name:
            continue

        match = re.search(r"\.layer\.(\d+)\.", name)
        if match is None:
            continue

        layer_index = int(match.group(1))
        candidates.append((name, layer_index))

    available_layers = sorted({layer for _, layer in candidates})
    if not available_layers:
        raise RuntimeError(
            "No compatible text-decoder query/value modules were found."
        )

    selected_layers = available_layers[-last_decoder_layers:]
    selected_names = [
        name
        for name, layer in candidates
        if layer in selected_layers
    ]

    if not selected_names:
        raise RuntimeError(
            "LoRA target discovery produced an empty module list."
        )

    return selected_names, selected_layers


def count_parameters(model: torch.nn.Module) -> dict[str, int | float]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    return {
        "total": int(total),
        "trainable": int(trainable),
        "trainable_percentage": round(
            trainable / total * 100,
            6,
        )
        if total
        else 0.0,
    }


def move_inputs(
    inputs: dict[str, torch.Tensor],
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, torch.Tensor]:
    moved: dict[str, torch.Tensor] = {}
    for name, tensor in inputs.items():
        tensor = tensor.to(device)
        if tensor.is_floating_point():
            tensor = tensor.to(dtype=dtype)
        moved[name] = tensor
    return moved


def prepare_sample(
    processor: Any,
    image_path: Path,
    question: str,
    answer: str,
    device: torch.device,
    dtype: torch.dtype,
    max_question_length: int,
    max_answer_length: int,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    with Image.open(image_path) as opened:
        image = opened.convert("RGB")

    inputs = processor(
        images=image,
        text=question,
        return_tensors="pt",
        truncation=True,
        max_length=max_question_length,
    )
    inputs = move_inputs(inputs, device, dtype)

    labels = processor.tokenizer(
        answer,
        return_tensors="pt",
        truncation=True,
        max_length=max_answer_length,
    ).input_ids.to(device)

    return inputs, labels


def save_logs(log_rows: list[dict[str, Any]], output_dir: Path) -> Path:
    path = output_dir / "training_log.csv"
    pd.DataFrame(log_rows).to_csv(
        path,
        index=False,
        encoding="utf-8-sig",
    )
    return path


def main() -> int:
    args = parse_args()

    if not args.csv.is_file():
        print(f"ERROR: CSV not found: {args.csv}", file=sys.stderr)
        return 2

    if not args.images_dir.is_dir():
        print(
            f"ERROR: images directory not found: {args.images_dir}",
            file=sys.stderr,
        )
        return 3

    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available.", file=sys.stderr)
        return 4

    if args.epochs < 1:
        raise ValueError("--epochs must be at least 1.")
    if args.gradient_accumulation < 1:
        raise ValueError("--gradient-accumulation must be at least 1.")
    if args.last_decoder_layers < 1:
        raise ValueError("--last-decoder-layers must be at least 1.")

    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    frame = pd.read_csv(args.csv, low_memory=False)
    required_columns = {"Figure_path", "Question", "Answer"}
    missing_columns = required_columns - set(frame.columns)
    if missing_columns:
        raise ValueError(
            f"CSV is missing columns: {sorted(missing_columns)}"
        )

    frame = frame.dropna(
        subset=["Figure_path", "Question", "Answer"]
    ).copy()
    frame["Figure_path"] = frame["Figure_path"].map(image_basename)
    frame["Question"] = frame["Question"].map(clean_text)
    frame["Answer"] = frame["Answer"].map(clean_text)
    frame = frame.sample(
        frac=1.0,
        random_state=args.seed,
    ).reset_index(drop=True)

    if args.limit > 0:
        frame = frame.head(args.limit).copy()

    missing_images = [
        image_name
        for image_name in frame["Figure_path"]
        if not (args.images_dir / image_name).is_file()
    ]
    if missing_images:
        print(
            f"ERROR: {len(missing_images)} referenced images are missing.",
            file=sys.stderr,
        )
        print(missing_images[:10], file=sys.stderr)
        return 5

    device = torch.device("cuda:0")
    dtype = torch.float16

    print(f"Training rows: {len(frame)}")
    print(f"Loading processor: {MODEL_ID}")
    processor = AutoProcessor.from_pretrained(MODEL_ID)

    print(f"Loading base model: {MODEL_ID}")
    base_model = BlipForQuestionAnswering.from_pretrained(
        MODEL_ID,
        low_cpu_mem_usage=True,
        torch_dtype=dtype,
    )

    if args.gradient_checkpointing:
        try:
            base_model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={
                    "use_reentrant": False,
                }
            )
            print("Gradient checkpointing enabled.")
        except TypeError:
            base_model.gradient_checkpointing_enable()
            print("Gradient checkpointing enabled with default settings.")

    target_modules, selected_layers = discover_lora_targets(
        base_model,
        last_decoder_layers=args.last_decoder_layers,
        attention_scope=args.attention_scope,
    )

    target_inventory_path = (
        args.output_dir / "lora_target_modules.json"
    )
    target_inventory_path.write_text(
        json.dumps(
            {
                "attention_scope": args.attention_scope,
                "selected_decoder_layers": selected_layers,
                "target_module_count": len(target_modules),
                "target_modules": target_modules,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        f"Selected decoder layers: {selected_layers} | "
        f"LoRA target modules: {len(target_modules)}"
    )

    lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.alpha,
        lora_dropout=args.dropout,
        bias="none",
        target_modules=target_modules,
    )

    model = get_peft_model(base_model, lora_config)
    # Keep the frozen base weights in FP16 while PEFT keeps trainable
    # adapter weights in FP32 for more stable optimization.
    model.to(device=device)
    model.train()

    parameter_counts = count_parameters(model)
    print(
        "Trainable parameters: "
        f"{parameter_counts['trainable']:,} / "
        f"{parameter_counts['total']:,} "
        f"({parameter_counts['trainable_percentage']:.6f}%)"
    )

    trainable_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    try:
        scaler = torch.amp.GradScaler(
            "cuda",
            enabled=True,
            init_scale=args.grad_scaler_init_scale,
            growth_interval=args.grad_scaler_growth_interval,
        )
    except TypeError:
        scaler = torch.cuda.amp.GradScaler(
            enabled=True,
            init_scale=args.grad_scaler_init_scale,
            growth_interval=args.grad_scaler_growth_interval,
        )

    total_micro_steps = len(frame) * args.epochs
    total_optimizer_steps = math.ceil(
        total_micro_steps / args.gradient_accumulation
    )

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    optimizer.zero_grad(set_to_none=True)

    log_rows: list[dict[str, Any]] = []
    loss_values: list[float] = []
    optimizer_step_attempts = 0
    optimizer_steps_applied = 0
    optimizer_steps_skipped = 0
    micro_step = 0
    started = time.perf_counter()
    status = "completed"
    error_message: str | None = None

    progress = tqdm(
        total=total_micro_steps,
        unit="sample",
        dynamic_ncols=True,
    )

    try:
        for epoch in range(1, args.epochs + 1):
            for row_index, row in frame.iterrows():
                micro_step += 1
                image_path = args.images_dir / row["Figure_path"]

                inputs, labels = prepare_sample(
                    processor=processor,
                    image_path=image_path,
                    question=row["Question"],
                    answer=row["Answer"],
                    device=device,
                    dtype=dtype,
                    max_question_length=args.max_question_length,
                    max_answer_length=args.max_answer_length,
                )

                with torch.autocast(
                    device_type="cuda",
                    dtype=dtype,
                ):
                    outputs = model(
                        **inputs,
                        labels=labels,
                    )
                    raw_loss = outputs.loss
                    scaled_loss = (
                        raw_loss / args.gradient_accumulation
                    )

                scaler.scale(scaled_loss).backward()
                loss_value = float(
                    raw_loss.detach().float().cpu().item()
                )
                loss_values.append(loss_value)

                should_step = (
                    micro_step % args.gradient_accumulation == 0
                    or micro_step == total_micro_steps
                )

                if should_step:
                    optimizer_step_attempts += 1
                    scaler.unscale_(optimizer)

                    gradient_norm_tensor = clip_grad_norm_(
                        trainable_parameters,
                        args.max_gradient_norm,
                    )
                    gradient_norm = float(
                        gradient_norm_tensor.detach().float().cpu().item()
                    )
                    gradient_norm_finite = math.isfinite(gradient_norm)

                    scale_before = float(scaler.get_scale())
                    scaler.step(optimizer)
                    scaler.update()
                    scale_after = float(scaler.get_scale())

                    step_skipped = scale_after < scale_before
                    if step_skipped:
                        optimizer_steps_skipped += 1
                    else:
                        optimizer_steps_applied += 1

                    optimizer.zero_grad(set_to_none=True)

                    recent_start = max(
                        0,
                        len(loss_values)
                        - args.gradient_accumulation,
                    )
                    recent_loss = statistics.fmean(
                        loss_values[recent_start:]
                    )

                    elapsed = time.perf_counter() - started
                    log_rows.append(
                        {
                            "epoch": epoch,
                            "micro_step": micro_step,
                            "optimizer_step_attempt": optimizer_step_attempts,
                            "optimizer_steps_applied": optimizer_steps_applied,
                            "optimizer_steps_skipped": optimizer_steps_skipped,
                            "step_skipped": step_skipped,
                            "loss": round(loss_value, 6),
                            "recent_mean_loss": round(
                                recent_loss,
                                6,
                            ),
                            "gradient_norm": (
                                round(gradient_norm, 6)
                                if gradient_norm_finite
                                else None
                            ),
                            "gradient_norm_finite": gradient_norm_finite,
                            "grad_scale_before": scale_before,
                            "grad_scale_after": scale_after,
                            "learning_rate": optimizer.param_groups[
                                0
                            ]["lr"],
                            "elapsed_seconds": round(elapsed, 4),
                            "peak_allocated_vram_mb": round(
                                torch.cuda.max_memory_allocated(
                                    device
                                )
                                / 1024**2,
                                2,
                            ),
                            "peak_reserved_vram_mb": round(
                                torch.cuda.max_memory_reserved(
                                    device
                                )
                                / 1024**2,
                                2,
                            ),
                        }
                    )

                progress.set_postfix(
                    loss=f"{loss_value:.4f}",
                    applied=f"{optimizer_steps_applied}/{total_optimizer_steps}",
                    skipped=optimizer_steps_skipped,
                )
                progress.update(1)

    except torch.cuda.OutOfMemoryError as exc:
        status = "cuda_out_of_memory"
        error_message = str(exc)
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
    except Exception as exc:  # noqa: BLE001
        status = "failed"
        error_message = f"{type(exc).__name__}: {exc}"
    finally:
        progress.close()

    elapsed_seconds = time.perf_counter() - started
    training_log_path = save_logs(log_rows, args.output_dir)

    adapter_dir = args.output_dir / "adapter"
    if status == "completed":
        model.save_pretrained(
            adapter_dir,
            safe_serialization=True,
        )
        processor.save_pretrained(adapter_dir)

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "error": error_message,
        "model_id": MODEL_ID,
        "source_csv": str(args.csv.resolve()),
        "images_directory": str(args.images_dir.resolve()),
        "output_directory": str(args.output_dir.resolve()),
        "device": {
            "name": torch.cuda.get_device_name(device),
            "total_vram_mb": round(
                torch.cuda.get_device_properties(device).total_memory
                / 1024**2,
                2,
            ),
        },
        "software": {
            "pytorch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
        },
        "training": {
            "rows": int(len(frame)),
            "epochs": args.epochs,
            "micro_steps_completed": micro_step,
            "optimizer_step_attempts": optimizer_step_attempts,
            "optimizer_steps_applied": optimizer_steps_applied,
            "optimizer_steps_skipped": optimizer_steps_skipped,
            "planned_optimizer_steps": total_optimizer_steps,
            "gradient_accumulation": args.gradient_accumulation,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "max_gradient_norm": args.max_gradient_norm,
            "seed": args.seed,
            "grad_scaler_init_scale": args.grad_scaler_init_scale,
            "grad_scaler_growth_interval": args.grad_scaler_growth_interval,
            "elapsed_seconds": round(elapsed_seconds, 4),
            "first_loss": round(loss_values[0], 6)
            if loss_values
            else None,
            "last_loss": round(loss_values[-1], 6)
            if loss_values
            else None,
            "mean_loss": round(
                statistics.fmean(loss_values),
                6,
            )
            if loss_values
            else None,
            "mean_first_20_losses": round(
                statistics.fmean(loss_values[:20]),
                6,
            )
            if loss_values
            else None,
            "mean_last_20_losses": round(
                statistics.fmean(loss_values[-20:]),
                6,
            )
            if loss_values
            else None,
        },
        "lora": {
            "rank": args.rank,
            "alpha": args.alpha,
            "dropout": args.dropout,
            "attention_scope": args.attention_scope,
            "last_decoder_layers": args.last_decoder_layers,
            "selected_decoder_layers": selected_layers,
            "target_module_count": len(target_modules),
        },
        "parameters": parameter_counts,
        "memory_mb": {
            "peak_allocated": round(
                torch.cuda.max_memory_allocated(device) / 1024**2,
                2,
            ),
            "peak_reserved": round(
                torch.cuda.max_memory_reserved(device) / 1024**2,
                2,
            ),
        },
        "artifacts": {
            "training_log": str(training_log_path.resolve()),
            "target_modules": str(
                target_inventory_path.resolve()
            ),
            "adapter": str(adapter_dir.resolve())
            if status == "completed"
            else None,
        },
    }

    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\nTraining finished")
    print(f"Status: {status}")
    print(
        "Optimizer steps applied: "
        f"{optimizer_steps_applied}/{total_optimizer_steps}"
    )
    print(f"Optimizer steps skipped: {optimizer_steps_skipped}")
    print(
        "Peak allocated VRAM: "
        f"{summary['memory_mb']['peak_allocated']} MB"
    )
    print(
        "Peak reserved VRAM: "
        f"{summary['memory_mb']['peak_reserved']} MB"
    )
    print(f"Summary: {summary_path.resolve()}")
    print(f"Training log: {training_log_path.resolve()}")

    if status == "completed":
        print(f"Adapter: {adapter_dir.resolve()}")
        return 0

    if status == "cuda_out_of_memory":
        print(
            "\nRetry suggestion: use --last-decoder-layers 1 "
            "--rank 2 --gradient-accumulation 16",
            file=sys.stderr,
        )
        return 6

    return 7


if __name__ == "__main__":
    raise SystemExit(main())
