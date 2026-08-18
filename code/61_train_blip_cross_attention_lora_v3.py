#!/usr/bin/env python3
"""Train Q/V LoRA on BLIP's image-question cross-attention branch.

Targets only the final N cross-attention layers of ``text_encoder`` by default.
This branch directly attends from question tokens to visual embeddings in BLIP VQA.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from common_vqa import (  # noqa: E402
    env_summary,
    get_answer_text,
    get_image_ref,
    get_options,
    get_question,
    index_images,
    read_csv_rows,
    resolve_image_path,
    write_json,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--project-root", type=Path, default=Path("."))
    p.add_argument("--data-root", type=Path, default=None)
    p.add_argument("--train-csv", type=Path, required=True)
    p.add_argument("--image-root", type=Path, required=True)
    p.add_argument("--model-name", default="Salesforce/blip-vqa-base")
    p.add_argument("--branch", choices=["text_encoder", "text_decoder"], default="text_encoder")
    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--alpha", type=int, default=16)
    p.add_argument("--dropout", type=float, default=0.05)
    p.add_argument("--last-n-layers", type=int, default=4)
    p.add_argument("--train-limit", type=int, default=2000)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--gradient-accumulation", type=int, default=8)
    p.add_argument("--max-question-length", type=int, default=32)
    p.add_argument("--max-answer-length", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--disable-amp", action="store_true")
    p.add_argument("--disable-gradient-checkpointing", action="store_true")
    p.add_argument("--local-files-only", action="store_true")
    p.add_argument("--out-dir", type=Path, required=True)
    return p.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def target_names(model: torch.nn.Module, branch: str, last_n_layers: int) -> List[str]:
    if not 1 <= last_n_layers <= 12:
        raise ValueError("--last-n-layers must be between 1 and 12")
    prefix = (
        "text_encoder.encoder.layer"
        if branch == "text_encoder"
        else "text_decoder.bert.encoder.layer"
    )
    first = 12 - last_n_layers
    wanted = sorted(
        f"{prefix}.{layer}.crossattention.self.{projection}"
        for layer in range(first, 12)
        for projection in ("query", "value")
    )
    existing = {
        name for name, module in model.named_modules()
        if isinstance(module, torch.nn.Linear)
    }
    missing = [name for name in wanted if name not in existing]
    if missing:
        raise RuntimeError("Cross-attention targets not found:\n" + "\n".join(missing))
    return wanted


def count_parameters(model: torch.nn.Module) -> Tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


class PMCDataset(Dataset):
    def __init__(
        self,
        samples: Sequence[Tuple[Path, str, str]],
        processor: Any,
        max_question_length: int,
        max_answer_length: int,
    ) -> None:
        self.samples = list(samples)
        self.processor = processor
        self.max_question_length = max_question_length
        self.max_answer_length = max_answer_length
        self.pad_token_id = processor.tokenizer.pad_token_id

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        image_path, question, answer = self.samples[idx]
        with Image.open(image_path) as image:
            encoded = self.processor(
                images=image.convert("RGB"),
                text=question,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=self.max_question_length,
            )
        answer_tokens = self.processor.tokenizer(
            answer,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_answer_length,
        )
        decoder_input_ids = answer_tokens.input_ids.squeeze(0)
        decoder_attention_mask = answer_tokens.attention_mask.squeeze(0)

        # In Transformers 5.x, BlipForQuestionAnswering uses `labels` as
        # decoder_input_ids when decoder_input_ids is omitted. Therefore,
        # labels containing -100 must never be passed without a separate,
        # valid decoder_input_ids tensor, or CUDA embedding lookup fails.
        labels = decoder_input_ids.clone()
        labels[decoder_attention_mask == 0] = -100

        item = {key: value.squeeze(0) for key, value in encoded.items()}
        item["decoder_input_ids"] = decoder_input_ids
        item["decoder_attention_mask"] = decoder_attention_mask
        item["labels"] = labels
        return item


def prepare_samples(
    rows: Sequence[Dict[str, str]],
    image_root: Path,
    limit: int,
) -> Tuple[List[Tuple[Path, str, str]], Dict[str, int]]:
    image_index = index_images([image_root])
    samples: List[Tuple[Path, str, str]] = []
    skipped: Dict[str, int] = {}
    for row in rows:
        if len(samples) >= limit:
            break
        try:
            question = get_question(row)
            options = get_options(row)
            answer = get_answer_text(row, options)
            image_ref = get_image_ref(row)
            image_path = resolve_image_path(image_ref, image_index, [image_root])
            if image_path is None:
                skipped["missing_image"] = skipped.get("missing_image", 0) + 1
                continue
            samples.append((image_path, question, answer))
        except Exception as exc:
            key = f"row_error:{type(exc).__name__}"
            skipped[key] = skipped.get(key, 0) + 1
    return samples, skipped


def main() -> int:
    args = parse_args()
    args.project_root = args.project_root.resolve()
    args.train_csv = args.train_csv.resolve()
    args.image_root = args.image_root.resolve()
    args.out_dir = args.out_dir.resolve()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required")
    device = torch.device("cuda")
    print(f"cuda_device={torch.cuda.get_device_name(0)}", flush=True)
    print(f"train_csv={args.train_csv}", flush=True)
    print(f"image_root={args.image_root}", flush=True)

    rows = read_csv_rows(args.train_csv)
    samples, skip_counts = prepare_samples(rows, args.image_root, args.train_limit)
    if not samples:
        raise RuntimeError(f"No valid training samples. skip_counts={skip_counts}")
    print(f"prepared_samples={len(samples)}", flush=True)
    print(f"skip_counts={skip_counts}", flush=True)

    from peft import LoraConfig, get_peft_model
    from transformers import AutoProcessor, BlipForQuestionAnswering

    processor = AutoProcessor.from_pretrained(
        args.model_name, local_files_only=args.local_files_only
    )
    base_model = BlipForQuestionAnswering.from_pretrained(
        args.model_name, local_files_only=args.local_files_only
    )
    targets = target_names(base_model, args.branch, args.last_n_layers)
    print(f"cross_attention_branch={args.branch}", flush=True)
    print(f"target_module_count={len(targets)}", flush=True)
    for name in targets:
        print(f"target={name}", flush=True)

    config = LoraConfig(
        r=args.rank,
        lora_alpha=args.alpha,
        lora_dropout=args.dropout,
        target_modules=targets,
        bias="none",
    )
    model = get_peft_model(base_model, config)

    if not args.disable_gradient_checkpointing:
        try:
            # Non-reentrant checkpointing does not require frozen input embeddings
            # to be marked trainable. That avoids BLIP's in-place positional-embedding
            # error while preserving gradients for the injected LoRA modules.
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
            print("gradient_checkpointing=enabled_non_reentrant", flush=True)
        except TypeError:
            # Compatibility fallback for older Transformers versions.
            model.gradient_checkpointing_enable()
            print("gradient_checkpointing=enabled_legacy", flush=True)
        except Exception as exc:
            print(f"gradient_checkpointing_warning={exc}", flush=True)

    model.to(device)
    model.train()
    total_params, trainable_params = count_parameters(model)
    trainable_percent = 100.0 * trainable_params / total_params
    print(f"total_parameters={total_params}", flush=True)
    print(f"trainable_parameters={trainable_params}", flush=True)
    print(f"trainable_percent={trainable_percent:.8f}", flush=True)

    dataset = PMCDataset(
        samples,
        processor,
        args.max_question_length,
        args.max_answer_length,
    )
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=False,
        generator=generator,
    )
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    use_amp = not args.disable_amp
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    optimizer.zero_grad(set_to_none=True)

    history: List[Dict[str, Any]] = []
    started = time.time()
    for epoch in range(1, args.epochs + 1):
        epoch_started = time.time()
        running_loss = 0.0
        optimizer_steps = 0
        for batch_idx, batch in enumerate(loader, start=1):
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            try:
                with torch.amp.autocast("cuda", enabled=use_amp):
                    output = model(**batch)
                    loss = output.loss / args.gradient_accumulation
                scaler.scale(loss).backward()
            except RuntimeError as exc:
                if "out of memory" in str(exc).lower():
                    optimizer.zero_grad(set_to_none=True)
                    torch.cuda.empty_cache()
                    raise RuntimeError(
                        "CUDA out of memory; retry with fewer target layers or shorter sequences"
                    ) from exc
                raise

            running_loss += float(loss.detach().cpu()) * args.gradient_accumulation
            should_step = (
                batch_idx % args.gradient_accumulation == 0
                or batch_idx == len(loader)
            )
            if should_step:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0
                )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                optimizer_steps += 1

            if batch_idx == 1 or batch_idx % 50 == 0 or batch_idx == len(loader):
                print(
                    f"epoch={epoch}/{args.epochs} batch={batch_idx}/{len(loader)} "
                    f"avg_loss={running_loss / batch_idx:.6f} "
                    f"max_gpu_gb={torch.cuda.max_memory_allocated() / 1024**3:.3f}",
                    flush=True,
                )

        record = {
            "epoch": epoch,
            "mean_loss": running_loss / len(loader),
            "optimizer_steps": optimizer_steps,
            "seconds": time.time() - epoch_started,
        }
        history.append(record)
        print(f"epoch_summary={record}", flush=True)
        model.save_pretrained(args.out_dir / f"adapter_epoch_{epoch}")

    adapter_dir = args.out_dir / "adapter"
    model.save_pretrained(adapter_dir)
    processor.save_pretrained(args.out_dir / "processor")
    elapsed = time.time() - started
    peak_gpu_gb = torch.cuda.max_memory_allocated() / 1024**3

    summary = {
        "run_name": args.out_dir.name,
        "model_name": args.model_name,
        "train_csv": str(args.train_csv),
        "image_root": str(args.image_root),
        "prepared_samples": len(samples),
        "skip_counts": skip_counts,
        "cross_attention_branch": args.branch,
        "target_layers": list(range(12 - args.last_n_layers, 12)),
        "target_projections": ["query", "value"],
        "target_modules": targets,
        "target_module_count": len(targets),
        "rank": args.rank,
        "alpha": args.alpha,
        "dropout": args.dropout,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "gradient_accumulation": args.gradient_accumulation,
        "seed": args.seed,
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "trainable_percent": trainable_percent,
        "training_seconds": elapsed,
        "peak_gpu_memory_gb": peak_gpu_gb,
        "history": history,
        "adapter_dir": str(adapter_dir),
        "environment": env_summary(),
    }
    write_json(args.out_dir / "cross_attention_lora_training_summary.json", summary)
    (args.out_dir / "target_modules.txt").write_text(
        "\n".join(targets) + "\n", encoding="utf-8"
    )
    with (args.out_dir / "training_history.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as f:
        writer = csv.DictWriter(
            f, fieldnames=["epoch", "mean_loss", "optimizer_steps", "seconds"]
        )
        writer.writeheader()
        writer.writerows(history)

    print(f"training_seconds={elapsed:.2f}", flush=True)
    print(f"peak_gpu_memory_gb={peak_gpu_gb:.3f}", flush=True)
    print(f"adapter_saved_to={adapter_dir}", flush=True)
    print(
        f"summary_saved_to={args.out_dir / 'cross_attention_lora_training_summary.json'}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        raise SystemExit(130)
