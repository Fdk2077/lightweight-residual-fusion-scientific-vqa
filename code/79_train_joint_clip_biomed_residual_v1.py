#!/usr/bin/env python3
"""Train a joint semantic residual scorer on frozen CLIP and BiomedCLIP features.

The two encoders remain fully frozen. For every answer option, the shared scorer
receives:
  * BiomedCLIP image/text elementwise-product features (512);
  * OpenAI CLIP image/text elementwise-product features (512);
  * per-question standardized similarity for both models (2);
  * standardized top-2 confidence margin for both models (2).

A small shared MLP adds a residual to the canonical BiomedCLIP score. The final
layer is initialized to zero, so training starts exactly at frozen BiomedCLIP.
No option-position identifier is provided, and a permutation-equivariance audit
is run before the validation gate is accepted.

This script uses training and development data only. It must not receive the
holdout or canonical test CSV.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from common_vqa import (  # type: ignore
    UserFacingError,
    clean_options,
    discover_image_roots,
    get_answer_label,
    get_image_ref,
    get_options,
    get_question,
    index_images,
    log,
    read_csv_rows,
    resolve_image_path,
)

BIOMED_MODEL_ID = (
    "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-"
    "vit_base_patch16_224"
)
CLIP_MODEL_ID = "openai/clip-vit-base-patch16"
PROMPT_TEMPLATE = "Question: {question} Answer: {option}"
BIOMED_SCORE_PROTOCOL = "canonical_autocast_similarity_v1"
CLIP_SCORE_PROTOCOL = "canonical_transformers_autocast_similarity_v1"
DEFAULT_SEEDS = (42, 7, 123)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--project-root", type=Path, default=Path("."))
    p.add_argument("--data-root", type=Path, default=None)
    p.add_argument("--train-csv", type=Path, required=True)
    p.add_argument("--val-csv", type=Path, required=True)
    p.add_argument("--image-root", type=Path, default=None)
    p.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    p.add_argument("--context-length", type=int, default=256)
    p.add_argument("--train-limit", type=int, default=None)
    p.add_argument("--val-limit", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--hidden-dim", type=int, default=32)
    p.add_argument("--dropout", type=float, default=0.10)
    p.add_argument("--temperature", type=float, default=20.0)
    p.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    p.add_argument(
        "--min-ensemble-delta",
        type=float,
        default=0.005,
        help="Development gate: minimum ensemble gain over BiomedCLIP.",
    )
    p.add_argument(
        "--min-positive-seeds",
        type=int,
        default=2,
        help="Development gate: seeds that must beat frozen BiomedCLIP.",
    )
    p.add_argument("--local-files-only", action="store_true")
    p.add_argument("--rebuild-cache", action="store_true")
    p.add_argument("--progress-every", type=int, default=100)
    p.add_argument("--out-dir", type=Path, required=True)
    return p.parse_args()


def choose_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise UserFacingError("CUDA was requested but is not available.")
    return torch.device(name)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def safe_torch_load(path: Path) -> Dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


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


def move_tokens(tokens: Any, device: torch.device) -> Any:
    if hasattr(tokens, "to"):
        return tokens.to(device)
    if isinstance(tokens, dict):
        return {key: value.to(device) for key, value in tokens.items()}
    raise TypeError(f"Unsupported tokenizer output type: {type(tokens)}")


def move_mapping(
    values: Mapping[str, torch.Tensor],
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in values.items()}


def make_prompts(question: str, options: Sequence[str]) -> List[str]:
    return [
        PROMPT_TEMPLATE.format(
            question=question.strip(),
            option=option.strip(),
        )
        for option in options
    ]


def label_to_index(label: str, option_count: int = 4) -> int:
    label = label.strip().upper()
    if len(label) != 1 or not ("A" <= label <= "Z"):
        raise UserFacingError(f"Unsupported answer label: {label!r}")
    index = ord(label) - ord("A")
    if not 0 <= index < option_count:
        raise UserFacingError(
            f"Answer label {label!r} is outside {option_count} options."
        )
    return index


def parse_rows(
    csv_path: Path,
    limit: int | None,
    image_roots: Sequence[Path],
    image_index: Mapping[str, Path],
) -> List[Dict[str, Any]]:
    rows = read_csv_rows(csv_path)
    if limit is not None:
        if limit <= 0:
            raise UserFacingError("Split limit must be positive.")
        rows = rows[:limit]

    parsed: List[Dict[str, Any]] = []
    skipped: Dict[str, int] = {}
    for row_index, row in enumerate(rows):
        try:
            question = get_question(row)
            options = clean_options(get_options(row))
            if len(options) != 4:
                raise UserFacingError(
                    f"Expected exactly four options, got {len(options)}."
                )
            gold_label = get_answer_label(row, options)
            if gold_label is None:
                raise UserFacingError("Could not resolve gold answer.")
            image_ref = get_image_ref(row)
            image_path = resolve_image_path(
                image_ref,
                image_index,
                image_roots,
            )
            if image_path is None:
                raise UserFacingError("Missing image.")
            parsed.append(
                {
                    "row_index": row_index,
                    "image_ref": image_ref,
                    "image_path": str(image_path),
                    "question": question,
                    "options": options,
                    "gold_label": gold_label,
                    "gold_index": label_to_index(gold_label),
                }
            )
        except Exception as exc:
            key = f"{type(exc).__name__}: {str(exc)[:160]}"
            skipped[key] = skipped.get(key, 0) + 1

    if skipped:
        raise UserFacingError(
            f"Parsing skipped rows for {csv_path}: {skipped}. "
            "The protocol requires zero skipped rows."
        )
    if not parsed:
        raise UserFacingError(f"No valid rows in {csv_path}.")
    return parsed


def metadata_rows(parsed: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    outputs: List[Dict[str, Any]] = []
    for row in parsed:
        options = row["options"]
        outputs.append(
            {
                "row_index": int(row["row_index"]),
                "image_ref": row["image_ref"],
                "image_path": row["image_path"],
                "question": row["question"],
                "gold_label": row["gold_label"],
                "choice_A": options[0],
                "choice_B": options[1],
                "choice_C": options[2],
                "choice_D": options[3],
            }
        )
    return outputs


def biomed_encode_text(model: Any, tokens: Any) -> torch.Tensor:
    if isinstance(tokens, dict):
        features = model.encode_text(**tokens)
    else:
        features = model.encode_text(tokens)
    return features / features.norm(
        dim=-1,
        keepdim=True,
    ).clamp_min(1e-12)


@torch.inference_mode()
def extract_biomed_features(
    *,
    split_name: str,
    parsed: Sequence[Mapping[str, Any]],
    csv_path: Path,
    limit: int | None,
    cache_path: Path,
    rebuild_cache: bool,
    model: Any,
    preprocess: Any,
    tokenizer: Any,
    device: torch.device,
    context_length: int,
    progress_every: int,
) -> Dict[str, Any]:
    expected_meta = {
        "model_id": BIOMED_MODEL_ID,
        "prompt_template": PROMPT_TEMPLATE,
        "score_protocol": BIOMED_SCORE_PROTOCOL,
        "csv_path": str(csv_path.resolve()),
        "limit": limit,
        "context_length": context_length,
    }
    if cache_path.exists() and not rebuild_cache:
        cached = safe_torch_load(cache_path)
        if cached.get("meta") == expected_meta:
            log(f"loading_biomed_{split_name}_cache={cache_path}")
            return cached
        log(f"ignoring_stale_biomed_{split_name}_cache={cache_path}")

    pair_features: List[torch.Tensor] = []
    base_scores: List[torch.Tensor] = []
    gold_indices: List[int] = []
    started = time.time()
    use_amp = device.type == "cuda"

    for row in parsed:
        with Image.open(row["image_path"]) as raw:
            image_tensor = (
                preprocess(raw.convert("RGB"))
                .unsqueeze(0)
                .to(device)
            )
        prompts = make_prompts(row["question"], row["options"])
        tokens = tokenizer(prompts, context_length=context_length)
        tokens = move_tokens(tokens, device)

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16 if use_amp else torch.float32,
            enabled=use_amp,
        ):
            image_features = model.encode_image(image_tensor)
            image_features = image_features / image_features.norm(
                dim=-1,
                keepdim=True,
            ).clamp_min(1e-12)
            text_features = biomed_encode_text(model, tokens)
            scores = (image_features @ text_features.T).squeeze(0)

        image_features = image_features.float()
        text_features = text_features.float()
        pair = image_features.expand_as(text_features) * text_features

        pair_features.append(pair.cpu())
        base_scores.append(scores.float().cpu())
        gold_indices.append(int(row["gold_index"]))

        done = len(gold_indices)
        if done == 1 or done % progress_every == 0:
            pred = torch.stack(base_scores).argmax(dim=1)
            gold = torch.tensor(gold_indices, dtype=torch.long)
            acc = float(pred.eq(gold).float().mean().item())
            log(
                f"model=biomed split={split_name} processed={done} "
                f"elapsed_s={time.time()-started:.1f} accuracy={acc:.4f}"
            )

    payload = {
        "meta": expected_meta,
        "pair_features": torch.stack(pair_features).contiguous(),
        "base_scores": torch.stack(base_scores).contiguous(),
        "gold_indices": torch.tensor(gold_indices, dtype=torch.long),
        "rows": metadata_rows(parsed),
    }
    torch.save(payload, cache_path)
    log(f"wrote_biomed_{split_name}_cache={cache_path}")
    return payload


def extract_clip_tensor(output: Any) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    for name in ("pooler_output", "image_embeds", "text_embeds"):
        value = getattr(output, name, None)
        if isinstance(value, torch.Tensor):
            return value
    if isinstance(output, (tuple, list)) and output:
        if isinstance(output[0], torch.Tensor):
            return output[0]
    raise TypeError(
        f"Could not extract feature tensor from {type(output).__name__}."
    )


def normalize_clip_output(output: Any) -> torch.Tensor:
    features = extract_clip_tensor(output)
    return features / features.norm(
        dim=-1,
        keepdim=True,
    ).clamp_min(1e-12)


@torch.inference_mode()
def extract_clip_features(
    *,
    split_name: str,
    parsed: Sequence[Mapping[str, Any]],
    csv_path: Path,
    limit: int | None,
    cache_path: Path,
    rebuild_cache: bool,
    model: Any,
    image_processor: Any,
    tokenizer: Any,
    device: torch.device,
    context_length: int,
    progress_every: int,
) -> Dict[str, Any]:
    expected_meta = {
        "model_id": CLIP_MODEL_ID,
        "prompt_template": PROMPT_TEMPLATE,
        "score_protocol": CLIP_SCORE_PROTOCOL,
        "csv_path": str(csv_path.resolve()),
        "limit": limit,
        "context_length": context_length,
    }
    if cache_path.exists() and not rebuild_cache:
        cached = safe_torch_load(cache_path)
        if cached.get("meta") == expected_meta:
            log(f"loading_clip_{split_name}_cache={cache_path}")
            return cached
        log(f"ignoring_stale_clip_{split_name}_cache={cache_path}")

    pair_features: List[torch.Tensor] = []
    base_scores: List[torch.Tensor] = []
    gold_indices: List[int] = []
    started = time.time()
    use_amp = device.type == "cuda"

    for row in parsed:
        with Image.open(row["image_path"]) as raw:
            image_inputs = image_processor(
                images=raw.convert("RGB"),
                return_tensors="pt",
            )
        image_inputs = move_mapping(image_inputs, device)
        prompts = make_prompts(row["question"], row["options"])
        text_inputs = tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=context_length,
            return_tensors="pt",
        )
        text_inputs = move_mapping(text_inputs, device)

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16 if use_amp else torch.float32,
            enabled=use_amp,
        ):
            image_features = normalize_clip_output(
                model.get_image_features(**image_inputs)
            )
            text_features = normalize_clip_output(
                model.get_text_features(**text_inputs)
            )
            scores = (image_features @ text_features.T).squeeze(0)

        image_features = image_features.float()
        text_features = text_features.float()
        pair = image_features.expand_as(text_features) * text_features

        pair_features.append(pair.cpu())
        base_scores.append(scores.float().cpu())
        gold_indices.append(int(row["gold_index"]))

        done = len(gold_indices)
        if done == 1 or done % progress_every == 0:
            pred = torch.stack(base_scores).argmax(dim=1)
            gold = torch.tensor(gold_indices, dtype=torch.long)
            acc = float(pred.eq(gold).float().mean().item())
            log(
                f"model=clip split={split_name} processed={done} "
                f"elapsed_s={time.time()-started:.1f} accuracy={acc:.4f}"
            )

    payload = {
        "meta": expected_meta,
        "pair_features": torch.stack(pair_features).contiguous(),
        "base_scores": torch.stack(base_scores).contiguous(),
        "gold_indices": torch.tensor(gold_indices, dtype=torch.long),
        "rows": metadata_rows(parsed),
    }
    torch.save(payload, cache_path)
    log(f"wrote_clip_{split_name}_cache={cache_path}")
    return payload


def assert_aligned(
    biomed: Mapping[str, Any],
    clip: Mapping[str, Any],
    split_name: str,
) -> None:
    if not torch.equal(
        biomed["gold_indices"],
        clip["gold_indices"],
    ):
        raise UserFacingError(f"Gold mismatch in {split_name}.")
    if biomed["rows"] != clip["rows"]:
        raise UserFacingError(f"Metadata mismatch in {split_name}.")
    if biomed["pair_features"].shape != clip["pair_features"].shape:
        raise UserFacingError(
            f"Pair-feature shape mismatch in {split_name}: "
            f"{biomed['pair_features'].shape} vs "
            f"{clip['pair_features'].shape}."
        )


def option_zscores(scores: torch.Tensor) -> torch.Tensor:
    mean = scores.mean(dim=1, keepdim=True)
    std = scores.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-6)
    return (scores - mean) / std


def top2_margin(scores: torch.Tensor) -> torch.Tensor:
    top2 = torch.topk(scores, k=2, dim=1).values
    return top2[:, 0] - top2[:, 1]


@dataclass
class FeatureNormalizer:
    biomed_pair_mean: torch.Tensor
    biomed_pair_std: torch.Tensor
    clip_pair_mean: torch.Tensor
    clip_pair_std: torch.Tensor
    biomed_margin_mean: torch.Tensor
    biomed_margin_std: torch.Tensor
    clip_margin_mean: torch.Tensor
    clip_margin_std: torch.Tensor

    def to_dict(self) -> Dict[str, torch.Tensor]:
        return {
            key: value.detach().cpu()
            for key, value in self.__dict__.items()
        }

    @classmethod
    def from_dict(
        cls,
        values: Mapping[str, torch.Tensor],
    ) -> "FeatureNormalizer":
        return cls(**{key: value.float() for key, value in values.items()})


def fit_normalizer(
    biomed_pair: torch.Tensor,
    clip_pair: torch.Tensor,
    biomed_scores: torch.Tensor,
    clip_scores: torch.Tensor,
) -> FeatureNormalizer:
    flat_biomed = biomed_pair.reshape(-1, biomed_pair.shape[-1]).float()
    flat_clip = clip_pair.reshape(-1, clip_pair.shape[-1]).float()
    biomed_margin = top2_margin(biomed_scores.float())
    clip_margin = top2_margin(clip_scores.float())

    return FeatureNormalizer(
        biomed_pair_mean=flat_biomed.mean(dim=0),
        biomed_pair_std=flat_biomed.std(
            dim=0,
            unbiased=False,
        ).clamp_min(1e-6),
        clip_pair_mean=flat_clip.mean(dim=0),
        clip_pair_std=flat_clip.std(
            dim=0,
            unbiased=False,
        ).clamp_min(1e-6),
        biomed_margin_mean=biomed_margin.mean(),
        biomed_margin_std=biomed_margin.std(
            unbiased=False,
        ).clamp_min(1e-6),
        clip_margin_mean=clip_margin.mean(),
        clip_margin_std=clip_margin.std(
            unbiased=False,
        ).clamp_min(1e-6),
    )


def build_option_features(
    biomed_pair: torch.Tensor,
    clip_pair: torch.Tensor,
    biomed_scores: torch.Tensor,
    clip_scores: torch.Tensor,
    normalizer: FeatureNormalizer,
) -> torch.Tensor:
    biomed_pair_norm = (
        biomed_pair - normalizer.biomed_pair_mean
    ) / normalizer.biomed_pair_std
    clip_pair_norm = (
        clip_pair - normalizer.clip_pair_mean
    ) / normalizer.clip_pair_std

    biomed_z = option_zscores(biomed_scores).unsqueeze(-1)
    clip_z = option_zscores(clip_scores).unsqueeze(-1)

    biomed_margin = (
        (
            top2_margin(biomed_scores)
            - normalizer.biomed_margin_mean
        )
        / normalizer.biomed_margin_std
    ).view(-1, 1, 1).expand(-1, 4, 1)
    clip_margin = (
        (
            top2_margin(clip_scores)
            - normalizer.clip_margin_mean
        )
        / normalizer.clip_margin_std
    ).view(-1, 1, 1).expand(-1, 4, 1)

    return torch.cat(
        [
            biomed_pair_norm,
            clip_pair_norm,
            biomed_z,
            clip_z,
            biomed_margin,
            clip_margin,
        ],
        dim=-1,
    )


class JointResidualScorer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        dropout: float,
        temperature: float,
    ) -> None:
        super().__init__()
        self.temperature = float(temperature)
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        final = self.network[-1]
        assert isinstance(final, nn.Linear)
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)

    def forward(
        self,
        option_features: torch.Tensor,
        biomed_base_scores: torch.Tensor,
    ) -> torch.Tensor:
        residual = self.network(option_features).squeeze(-1)
        return self.temperature * (biomed_base_scores + residual)


@dataclass
class SeedResult:
    seed: int
    best_epoch: int
    best_val_accuracy: float
    best_val_loss: float
    train_accuracy_at_best: float
    state_path: str


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


@torch.inference_mode()
def evaluate_model(
    model: JointResidualScorer,
    *,
    biomed_pair: torch.Tensor,
    clip_pair: torch.Tensor,
    biomed_scores: torch.Tensor,
    clip_scores: torch.Tensor,
    gold: torch.Tensor,
    normalizer: FeatureNormalizer,
    device: torch.device,
    batch_size: int,
) -> Tuple[float, float, torch.Tensor]:
    model.eval()
    logits_parts: List[torch.Tensor] = []
    total_loss = 0.0
    total_rows = 0

    for start in range(0, gold.numel(), batch_size):
        end = min(start + batch_size, gold.numel())
        features = build_option_features(
            biomed_pair[start:end],
            clip_pair[start:end],
            biomed_scores[start:end],
            clip_scores[start:end],
            normalizer,
        ).to(device)
        base = biomed_scores[start:end].to(device)
        target = gold[start:end].to(device)
        logits = model(features, base)
        loss = nn.functional.cross_entropy(
            logits,
            target,
            reduction="sum",
        )
        total_loss += float(loss.item())
        total_rows += int(target.numel())
        logits_parts.append(logits.detach().cpu())

    all_logits = torch.cat(logits_parts, dim=0)
    accuracy = float(
        all_logits.argmax(dim=1).eq(gold).float().mean().item()
    )
    return accuracy, total_loss / total_rows, all_logits


def train_one_seed(
    *,
    seed: int,
    train_biomed: Mapping[str, Any],
    train_clip: Mapping[str, Any],
    val_biomed: Mapping[str, Any],
    val_clip: Mapping[str, Any],
    normalizer: FeatureNormalizer,
    device: torch.device,
    batch_size: int,
    epochs: int,
    patience: int,
    learning_rate: float,
    weight_decay: float,
    hidden_dim: int,
    dropout: float,
    temperature: float,
    out_dir: Path,
) -> SeedResult:
    set_seed(seed)

    train_biomed_pair = train_biomed["pair_features"].float()
    train_clip_pair = train_clip["pair_features"].float()
    train_biomed_scores = train_biomed["base_scores"].float()
    train_clip_scores = train_clip["base_scores"].float()
    train_gold = train_biomed["gold_indices"].long()

    val_biomed_pair = val_biomed["pair_features"].float()
    val_clip_pair = val_clip["pair_features"].float()
    val_biomed_scores = val_biomed["base_scores"].float()
    val_clip_scores = val_clip["base_scores"].float()
    val_gold = val_biomed["gold_indices"].long()

    input_dim = (
        train_biomed_pair.shape[-1]
        + train_clip_pair.shape[-1]
        + 4
    )
    model = JointResidualScorer(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        dropout=dropout,
        temperature=temperature,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    generator = torch.Generator()
    generator.manual_seed(seed)
    loader = DataLoader(
        TensorDataset(torch.arange(train_gold.numel())),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
        drop_last=False,
    )

    best_accuracy = -1.0
    best_loss = math.inf
    best_epoch = 0
    best_train_accuracy = 0.0
    best_state: Dict[str, torch.Tensor] | None = None
    stale_epochs = 0
    history: List[Dict[str, Any]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_rows = 0
        total_correct = 0

        for (indices,) in loader:
            features = build_option_features(
                train_biomed_pair[indices],
                train_clip_pair[indices],
                train_biomed_scores[indices],
                train_clip_scores[indices],
                normalizer,
            ).to(device)
            base = train_biomed_scores[indices].to(device)
            target = train_gold[indices].to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(features, base)
            loss = nn.functional.cross_entropy(logits, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=5.0,
            )
            optimizer.step()

            total_loss += float(loss.item()) * target.numel()
            total_rows += int(target.numel())
            total_correct += int(
                logits.argmax(dim=1).eq(target).sum().item()
            )

        train_loss = total_loss / total_rows
        train_accuracy = total_correct / total_rows
        val_accuracy, val_loss, _ = evaluate_model(
            model,
            biomed_pair=val_biomed_pair,
            clip_pair=val_clip_pair,
            biomed_scores=val_biomed_scores,
            clip_scores=val_clip_scores,
            gold=val_gold,
            normalizer=normalizer,
            device=device,
            batch_size=batch_size,
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
            }
        )

        improved = (
            val_accuracy > best_accuracy + 1e-12
            or (
                abs(val_accuracy - best_accuracy) <= 1e-12
                and val_loss < best_loss - 1e-9
            )
        )
        if improved:
            best_accuracy = val_accuracy
            best_loss = val_loss
            best_epoch = epoch
            best_train_accuracy = train_accuracy
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1

        if epoch == 1 or epoch % 5 == 0 or improved:
            log(
                f"seed={seed} epoch={epoch} "
                f"train_acc={train_accuracy:.4f} "
                f"val_acc={val_accuracy:.4f} "
                f"val_loss={val_loss:.4f} "
                f"best={best_accuracy:.4f}@{best_epoch}"
            )
        if stale_epochs >= patience:
            log(f"seed={seed} early_stop_epoch={epoch}")
            break

    if best_state is None:
        raise RuntimeError("No best state recorded.")

    seed_dir = out_dir / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    state_path = seed_dir / "best_state.pt"
    torch.save(
        {
            "experiment": "joint_clip_biomed_semantic_residual_v1",
            "biomed_model_id": BIOMED_MODEL_ID,
            "clip_model_id": CLIP_MODEL_ID,
            "prompt_template": PROMPT_TEMPLATE,
            "input_dim": int(input_dim),
            "hidden_dim": int(hidden_dim),
            "dropout": float(dropout),
            "temperature": float(temperature),
            "seed": int(seed),
            "best_epoch": int(best_epoch),
            "normalizer": normalizer.to_dict(),
            "state_dict": best_state,
        },
        state_path,
    )
    write_json(
        seed_dir / "history.json",
        {"seed": seed, "history": history},
    )

    return SeedResult(
        seed=seed,
        best_epoch=best_epoch,
        best_val_accuracy=best_accuracy,
        best_val_loss=best_loss,
        train_accuracy_at_best=best_train_accuracy,
        state_path=str(state_path),
    )


def load_seed_model(
    state_path: Path,
    device: torch.device,
) -> Tuple[JointResidualScorer, FeatureNormalizer]:
    payload = safe_torch_load(state_path)
    model = JointResidualScorer(
        input_dim=int(payload["input_dim"]),
        hidden_dim=int(payload["hidden_dim"]),
        dropout=float(payload["dropout"]),
        temperature=float(payload["temperature"]),
    ).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    normalizer = FeatureNormalizer.from_dict(payload["normalizer"])
    return model, normalizer


@torch.inference_mode()
def ensemble_logits(
    state_paths: Sequence[Path],
    *,
    biomed: Mapping[str, Any],
    clip: Mapping[str, Any],
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    logits: List[torch.Tensor] = []
    for state_path in state_paths:
        model, normalizer = load_seed_model(state_path, device)
        _, _, seed_logits = evaluate_model(
            model,
            biomed_pair=biomed["pair_features"].float(),
            clip_pair=clip["pair_features"].float(),
            biomed_scores=biomed["base_scores"].float(),
            clip_scores=clip["base_scores"].float(),
            gold=biomed["gold_indices"].long(),
            normalizer=normalizer,
            device=device,
            batch_size=batch_size,
        )
        logits.append(seed_logits)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return torch.stack(logits).mean(dim=0)


@torch.inference_mode()
def permutation_equivariance_audit(
    state_paths: Sequence[Path],
    *,
    biomed: Mapping[str, Any],
    clip: Mapping[str, Any],
    device: torch.device,
    batch_size: int,
) -> Dict[str, Any]:
    original_logits = ensemble_logits(
        state_paths,
        biomed=biomed,
        clip=clip,
        device=device,
        batch_size=batch_size,
    )
    original_pred = original_logits.argmax(dim=1)

    generator = torch.Generator()
    generator.manual_seed(20260728)
    permutations = torch.stack(
        [torch.randperm(4, generator=generator) for _ in range(original_pred.numel())]
    )

    row_indices = torch.arange(original_pred.numel()).unsqueeze(1)
    permuted_biomed = {
        **biomed,
        "pair_features": biomed["pair_features"][row_indices, permutations],
        "base_scores": biomed["base_scores"][row_indices, permutations],
    }
    permuted_clip = {
        **clip,
        "pair_features": clip["pair_features"][row_indices, permutations],
        "base_scores": clip["base_scores"][row_indices, permutations],
    }

    permuted_logits = ensemble_logits(
        state_paths,
        biomed=permuted_biomed,
        clip=permuted_clip,
        device=device,
        batch_size=batch_size,
    )
    permuted_position = permuted_logits.argmax(dim=1)
    mapped_back = permutations[
        torch.arange(permuted_position.numel()),
        permuted_position,
    ]
    mismatches = int(mapped_back.ne(original_pred).sum().item())
    return {
        "rows": int(original_pred.numel()),
        "permutation_seed": 20260728,
        "prediction_mismatches_after_unpermute": mismatches,
        "passed": mismatches == 0,
    }


def main() -> int:
    args = parse_args()
    args.project_root = args.project_root.resolve()
    args.data_root = args.data_root.resolve() if args.data_root else None
    args.train_csv = args.train_csv.resolve()
    args.val_csv = args.val_csv.resolve()
    args.image_root = args.image_root.resolve() if args.image_root else None
    args.out_dir = args.out_dir.resolve()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.train_csv == args.val_csv:
        raise UserFacingError("Training and validation CSVs must differ.")
    if args.hidden_dim <= 0:
        raise UserFacingError("--hidden-dim must be positive.")
    if not 0.0 <= args.dropout < 1.0:
        raise UserFacingError("--dropout must be in [0, 1).")
    if not args.seeds:
        raise UserFacingError("At least one seed is required.")

    device = choose_device(args.device)
    log(f"device={device}")
    if device.type == "cuda":
        log(f"cuda_device={torch.cuda.get_device_name(0)}")

    image_roots = (
        [args.image_root]
        if args.image_root
        else discover_image_roots(args.project_root, args.data_root)
    )
    image_roots = [
        path.resolve()
        for path in image_roots
        if path and path.is_dir()
    ]
    if not image_roots:
        raise UserFacingError("No image root found. Pass --image-root.")
    image_index = index_images(image_roots)
    log(f"indexed_image_keys={len(image_index)}")

    train_rows = parse_rows(
        args.train_csv,
        args.train_limit,
        image_roots,
        image_index,
    )
    val_rows = parse_rows(
        args.val_csv,
        args.val_limit,
        image_roots,
        image_index,
    )

    try:
        from open_clip import create_model_from_pretrained, get_tokenizer
    except ImportError as exc:
        raise UserFacingError("open_clip is not installed.") from exc

    log(f"loading_biomed_model={BIOMED_MODEL_ID}")
    biomed_model, biomed_preprocess = create_model_from_pretrained(
        BIOMED_MODEL_ID
    )
    biomed_tokenizer = get_tokenizer(BIOMED_MODEL_ID)
    biomed_model = biomed_model.to(device)
    biomed_model.eval()
    for parameter in biomed_model.parameters():
        parameter.requires_grad_(False)

    train_biomed = extract_biomed_features(
        split_name="train",
        parsed=train_rows,
        csv_path=args.train_csv,
        limit=args.train_limit,
        cache_path=args.out_dir / "train_biomed_features.pt",
        rebuild_cache=args.rebuild_cache,
        model=biomed_model,
        preprocess=biomed_preprocess,
        tokenizer=biomed_tokenizer,
        device=device,
        context_length=args.context_length,
        progress_every=args.progress_every,
    )
    val_biomed = extract_biomed_features(
        split_name="val",
        parsed=val_rows,
        csv_path=args.val_csv,
        limit=args.val_limit,
        cache_path=args.out_dir / "val_biomed_features.pt",
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
        from transformers import (
            AutoImageProcessor,
            AutoTokenizer,
            CLIPModel,
        )
    except ImportError as exc:
        raise UserFacingError("transformers is not installed.") from exc

    log(f"loading_clip_model={CLIP_MODEL_ID}")
    clip_model = CLIPModel.from_pretrained(
        CLIP_MODEL_ID,
        local_files_only=args.local_files_only,
    )
    clip_tokenizer = AutoTokenizer.from_pretrained(
        CLIP_MODEL_ID,
        local_files_only=args.local_files_only,
    )
    clip_image_processor = AutoImageProcessor.from_pretrained(
        CLIP_MODEL_ID,
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

    train_clip = extract_clip_features(
        split_name="train",
        parsed=train_rows,
        csv_path=args.train_csv,
        limit=args.train_limit,
        cache_path=args.out_dir / "train_clip_features.pt",
        rebuild_cache=args.rebuild_cache,
        model=clip_model,
        image_processor=clip_image_processor,
        tokenizer=clip_tokenizer,
        device=device,
        context_length=clip_context_length,
        progress_every=args.progress_every,
    )
    val_clip = extract_clip_features(
        split_name="val",
        parsed=val_rows,
        csv_path=args.val_csv,
        limit=args.val_limit,
        cache_path=args.out_dir / "val_clip_features.pt",
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

    assert_aligned(train_biomed, train_clip, "train")
    assert_aligned(val_biomed, val_clip, "val")

    normalizer = fit_normalizer(
        train_biomed["pair_features"].float(),
        train_clip["pair_features"].float(),
        train_biomed["base_scores"].float(),
        train_clip["base_scores"].float(),
    )

    val_gold = val_biomed["gold_indices"].long()
    val_base_scores = val_biomed["base_scores"].float()
    base_val_pred = val_base_scores.argmax(dim=1)
    base_val_accuracy = float(
        base_val_pred.eq(val_gold).float().mean().item()
    )
    log(f"frozen_biomedclip_val_accuracy={base_val_accuracy:.4f}")

    input_dim = int(
        train_biomed["pair_features"].shape[-1]
        + train_clip["pair_features"].shape[-1]
        + 4
    )
    parameter_probe = JointResidualScorer(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        temperature=args.temperature,
    )
    trainable_parameters = count_trainable_parameters(parameter_probe)
    del parameter_probe
    log(f"joint_scorer_trainable_parameters={trainable_parameters}")

    seed_results: List[SeedResult] = []
    state_paths: List[Path] = []
    for seed in args.seeds:
        result = train_one_seed(
            seed=seed,
            train_biomed=train_biomed,
            train_clip=train_clip,
            val_biomed=val_biomed,
            val_clip=val_clip,
            normalizer=normalizer,
            device=device,
            batch_size=args.batch_size,
            epochs=args.epochs,
            patience=args.patience,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            hidden_dim=args.hidden_dim,
            dropout=args.dropout,
            temperature=args.temperature,
            out_dir=args.out_dir,
        )
        seed_results.append(result)
        state_paths.append(Path(result.state_path))

    ensemble_val_logits = ensemble_logits(
        state_paths,
        biomed=val_biomed,
        clip=val_clip,
        device=device,
        batch_size=args.batch_size,
    )
    ensemble_val_pred = ensemble_val_logits.argmax(dim=1)
    ensemble_val_accuracy = float(
        ensemble_val_pred.eq(val_gold).float().mean().item()
    )
    ensemble_delta = ensemble_val_accuracy - base_val_accuracy

    seed_accuracies = [result.best_val_accuracy for result in seed_results]
    mean_seed_accuracy = float(np.mean(seed_accuracies))
    std_seed_accuracy = (
        float(np.std(seed_accuracies, ddof=1))
        if len(seed_accuracies) > 1
        else 0.0
    )
    positive_seeds = sum(
        accuracy > base_val_accuracy
        for accuracy in seed_accuracies
    )

    permutation_audit = permutation_equivariance_audit(
        state_paths,
        biomed=val_biomed,
        clip=val_clip,
        device=device,
        batch_size=args.batch_size,
    )

    epsilon = 1e-8
    gate_pass = (
        ensemble_delta + epsilon >= args.min_ensemble_delta
        and positive_seeds >= args.min_positive_seeds
        and permutation_audit["passed"]
    )

    manifest_path = args.out_dir / "ensemble_manifest.pt"
    torch.save(
        {
            "experiment": "joint_clip_biomed_semantic_residual_v1",
            "biomed_model_id": BIOMED_MODEL_ID,
            "clip_model_id": CLIP_MODEL_ID,
            "prompt_template": PROMPT_TEMPLATE,
            "biomed_score_protocol": BIOMED_SCORE_PROTOCOL,
            "clip_score_protocol": CLIP_SCORE_PROTOCOL,
            "input_dim": input_dim,
            "hidden_dim": args.hidden_dim,
            "dropout": args.dropout,
            "temperature": args.temperature,
            "seeds": list(args.seeds),
            "state_paths": [str(path.resolve()) for path in state_paths],
            "validation_gate_pass": gate_pass,
            "validation_baseline_accuracy": base_val_accuracy,
            "validation_ensemble_accuracy": ensemble_val_accuracy,
            "validation_ensemble_delta": ensemble_delta,
        },
        manifest_path,
    )

    prediction_rows: List[Dict[str, Any]] = []
    for index, row in enumerate(val_biomed["rows"]):
        baseline_index = int(base_val_pred[index].item())
        ensemble_index = int(ensemble_val_pred[index].item())
        prediction_rows.append(
            {
                **row,
                "baseline_pred_label": chr(ord("A") + baseline_index),
                "baseline_correct": int(baseline_index == int(val_gold[index])),
                "ensemble_pred_label": chr(ord("A") + ensemble_index),
                "ensemble_correct": int(ensemble_index == int(val_gold[index])),
                "ensemble_logits_json": json.dumps(
                    ensemble_val_logits[index].tolist()
                ),
            }
        )
    write_csv(
        args.out_dir / "ensemble_val_predictions.csv",
        prediction_rows,
    )

    summary = {
        "experiment": "joint_clip_biomed_semantic_residual_v1",
        "status": "exploratory_after_prior_validation_experiments",
        "biomed_model_id": BIOMED_MODEL_ID,
        "clip_model_id": CLIP_MODEL_ID,
        "prompt_template": PROMPT_TEMPLATE,
        "train_csv": str(args.train_csv),
        "val_csv": str(args.val_csv),
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
        "embedding_dim_each_model": int(
            train_biomed["pair_features"].shape[-1]
        ),
        "input_dim": input_dim,
        "hidden_dim": args.hidden_dim,
        "trainable_parameters": trainable_parameters,
        "hyperparameters": {
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "patience": args.patience,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "dropout": args.dropout,
            "temperature": args.temperature,
            "seeds": list(args.seeds),
        },
        "frozen_biomedclip_val_accuracy": base_val_accuracy,
        "seed_results": [result.__dict__ for result in seed_results],
        "mean_seed_val_accuracy": mean_seed_accuracy,
        "std_seed_val_accuracy": std_seed_accuracy,
        "positive_seeds": positive_seeds,
        "ensemble_val_accuracy": ensemble_val_accuracy,
        "ensemble_delta": ensemble_delta,
        "permutation_equivariance_audit": permutation_audit,
        "development_gate": {
            "min_ensemble_delta": args.min_ensemble_delta,
            "min_positive_seeds": args.min_positive_seeds,
            "numeric_epsilon": epsilon,
            "passed": gate_pass,
        },
        "ensemble_manifest": str(manifest_path),
        "next_action": (
            "Evaluate the frozen three-seed ensemble once on the clean "
            "group-disjoint 290-row holdout."
            if gate_pass
            else "Stop. Do not evaluate the holdout for this scorer."
        ),
    }
    write_json(args.out_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
