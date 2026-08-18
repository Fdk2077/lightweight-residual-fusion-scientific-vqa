#!/usr/bin/env python3
"""One-shot temporal evaluation of frozen PMC-VQA models.

The script consumes the PASS_FOR_E4B audit produced by E4A, constructs a clean
post-cutoff evaluation set, extracts only the required images from the original
PMC-VQA image archive, and evaluates four frozen systems:

1. BiomedCLIP zero-shot baseline;
2. OpenAI CLIP zero-shot baseline;
3. the prespecified three-seed nonlinear joint residual scorer (primary);
4. the post-hoc 1,029-parameter linear ablation (secondary/exploratory).

The feature extraction cache is chunked and resumable. No model is trained or
selected on the temporal set. The final run refuses to overwrite a completed
result unless --allow-overwrite is supplied.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import re
import shutil
import sys
import time
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from statistics import NormalDist

SCRIPT_DIR = Path(__file__).resolve().parent
TRAIN_SCRIPT = SCRIPT_DIR / "79_train_joint_clip_biomed_residual_v1.py"
VALID_LABELS = ("A", "B", "C", "D")
FINAL_SUMMARY_NAME = "temporal_evaluation_summary.json"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate frozen CLIP/BiomedCLIP residual scorers on the E4A temporal set."
    )
    p.add_argument("--project-root", type=Path, default=Path("."))
    p.add_argument("--audit-dir", type=Path, required=True)
    p.add_argument("--image-archive", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--approved-manifest", type=Path, default=None)
    p.add_argument("--linear-states", type=Path, default=None)
    p.add_argument("--train-feature-root", type=Path, default=None)
    p.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    p.add_argument("--context-length", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--chunk-size", type=int, default=200)
    p.add_argument("--progress-every", type=int, default=50)
    p.add_argument("--bootstrap-repeats", type=int, default=20000)
    p.add_argument("--bootstrap-seed", type=int, default=20260801)
    p.add_argument("--eval-limit", type=int, default=None,
                   help="Optional smoke-test limit. Omit for the final one-shot run.")
    p.add_argument("--prepare-only", action="store_true")
    p.add_argument("--local-files-only", action="store_true")
    p.add_argument("--rebuild-cache", action="store_true")
    p.add_argument("--allow-overwrite", action="store_true")
    return p.parse_args()


def log(message: str) -> None:
    print(f"[E4B] {message}", flush=True)


def scalar_to_text(value: Any) -> str:
    if value is None or value is pd.NA:
        return ""
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        missing = False
    if isinstance(missing, (bool, np.bool_)) and bool(missing):
        return ""
    return str(value)


def normalize_text(value: Any) -> str:
    return re.sub(r"\s+", " ", scalar_to_text(value).strip().lower())


def normalize_choice_text(value: Any, expected_label: str) -> str:
    """Remove only A:/B:/C:/D: style metadata prefixes.

    A hyphen is deliberately not a prefix separator because biomedical terms
    such as B-mode are legitimate option text.
    """
    raw = scalar_to_text(value).strip()
    raw = re.sub(
        rf"^\s*{re.escape(expected_label)}\s*[:.)]\s*",
        "",
        raw,
        count=1,
        flags=re.IGNORECASE,
    )
    return normalize_text(raw)


def normalize_archive_member(value: Any, expected_archive: Path | None = None) -> str:
    """Return the internal ZIP member from either `member` or `archive::member`.

    E4A records image locators as:
        E:\\path\\images.zip::images/PMC...jpg
    while zipfile expects only:
        images/PMC...jpg
    """
    raw = scalar_to_text(value).strip()
    if not raw:
        return ""

    if "::" in raw:
        archive_part, member_part = raw.rsplit("::", 1)
        if expected_archive is not None and archive_part.strip():
            try:
                recorded = Path(archive_part).resolve()
                expected = Path(expected_archive).resolve()
                if recorded != expected:
                    log(
                        "warning: manifest archive path differs from --image-archive; "
                        f"manifest={recorded} supplied={expected}. "
                        "The supplied archive will be used."
                    )
            except (OSError, RuntimeError, ValueError):
                pass
        raw = member_part.strip()

    # ZIP members always use forward slashes and are relative to the archive root.
    raw = raw.replace("\\\\", "/").lstrip("/")
    while raw.startswith("./"):
        raw = raw[2:]
    return raw


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise RuntimeError(f"Refusing to write an empty CSV: {path}")
    fields: List[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                fields.append(key)
                seen.add(key)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def load_training_module() -> Any:
    if not TRAIN_SCRIPT.is_file():
        raise FileNotFoundError(
            f"Required sibling script not found: {TRAIN_SCRIPT}. "
            "Place E4B beside script 79 in scripts/next_experiments."
        )
    spec = importlib.util.spec_from_file_location("joint_train_e4b", TRAIN_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {TRAIN_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def choose_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    return torch.device(name)


def resolve_defaults(args: argparse.Namespace) -> None:
    args.project_root = args.project_root.resolve()
    args.audit_dir = args.audit_dir.resolve()
    args.image_archive = args.image_archive.resolve()
    args.out_dir = args.out_dir.resolve()

    if args.approved_manifest is None:
        args.approved_manifest = (
            args.project_root
            / "outputs/joint_clip_biomed/joint_holdout290_ensemble_v1/"
              "ensemble_manifest_holdout_approved.pt"
        )
    if args.linear_states is None:
        args.linear_states = (
            args.project_root
            / "outputs/paper_e1_gpu_v1_p1/E1_linear/ensemble_states.pt"
        )
    if args.train_feature_root is None:
        args.train_feature_root = (
            args.project_root
            / "outputs/joint_clip_biomed/"
              "joint_train12000_dev1000_multiseed_v1"
        )

    args.approved_manifest = args.approved_manifest.resolve()
    args.linear_states = args.linear_states.resolve()
    args.train_feature_root = args.train_feature_root.resolve()


def validate_inputs(args: argparse.Namespace) -> Dict[str, Any]:
    if args.chunk_size <= 0 or args.batch_size <= 0:
        raise ValueError("--chunk-size and --batch-size must be positive.")
    if args.eval_limit is not None and args.eval_limit <= 0:
        raise ValueError("--eval-limit must be positive.")
    required = [
        args.audit_dir / "temporal_audit_summary.json",
        args.audit_dir / "temporal_candidate_manifest_all.csv",
        args.image_archive,
        args.approved_manifest,
        args.linear_states,
        args.train_feature_root / "train_biomed_features.pt",
        args.train_feature_root / "train_clip_features.pt",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Required files are missing: {missing}")

    audit_summary = json.loads(
        (args.audit_dir / "temporal_audit_summary.json").read_text(encoding="utf-8")
    )
    gate = audit_summary.get("e4b_gate", {})
    if gate.get("decision") != "PASS_FOR_E4B" or gate.get("failed"):
        raise RuntimeError(
            "E4A did not produce a clean PASS_FOR_E4B decision. "
            f"Observed gate: {gate}"
        )
    temporal = audit_summary.get("temporal_candidates", {})
    if int(temporal.get("pmcid_overlap_with_existing_splits", -1)) != 0:
        raise RuntimeError("E4A reports PMCID overlap with existing splits.")
    if int(temporal.get("image_overlap_with_existing_splits", -1)) != 0:
        raise RuntimeError("E4A reports image overlap with existing splits.")
    return audit_summary


def prepare_temporal_dataset(
    args: argparse.Namespace,
    audit_summary: Mapping[str, Any],
) -> Tuple[Path, Path, pd.DataFrame, Dict[str, Any]]:
    source_path = args.audit_dir / "temporal_candidate_manifest_all.csv"
    df = pd.read_csv(source_path)
    if "image_archive_member" in df.columns:
        df["image_archive_locator_original"] = df["image_archive_member"]
        df["image_archive_member"] = df["image_archive_member"].map(
            lambda value: normalize_archive_member(value, args.image_archive)
        )
    required_columns = {
        "candidate_id", "source_file", "source_row_index", "Figure_path",
        "image_ref_norm", "image_archive_member", "image_available", "pmcid",
        "date_lower_bound", "date_upper_bound", "Question",
        "Choice A", "Choice B", "Choice C", "Choice D", "Answer",
        "Answer_label",
    }
    missing = sorted(required_columns.difference(df.columns))
    if missing:
        raise RuntimeError(f"Temporal manifest is missing columns: {missing}")

    labels = df["Answer_label"].map(normalize_text).str.upper()
    label_valid = labels.isin(VALID_LABELS)
    choice_unique: List[bool] = []
    answer_match: List[bool] = []
    for index, row in df.iterrows():
        label = labels.iloc[index]
        if label not in VALID_LABELS:
            choice_unique.append(False)
            answer_match.append(False)
            continue
        choices = [
            normalize_choice_text(row[f"Choice {candidate}"], candidate)
            for candidate in VALID_LABELS
        ]
        choice_unique.append(all(choices) and len(set(choices)) == 4)
        answer_norm = normalize_choice_text(row["Answer"], label)
        gold_norm = normalize_choice_text(row[f"Choice {label}"], label)
        answer_as_label = normalize_text(row["Answer"]).upper()
        answer_match.append(
            (answer_norm != "" and answer_norm == gold_norm)
            or answer_as_label == label
        )

    df = df.assign(
        _label=labels,
        _choice_unique=np.asarray(choice_unique, dtype=bool),
        _answer_match=np.asarray(answer_match, dtype=bool),
        _image_available=df["image_available"].astype(bool),
        _member_present=df["image_archive_member"].map(normalize_text).ne(""),
    )
    df["_row_valid"] = (
        label_valid
        & df["_choice_unique"]
        & df["_answer_match"]
        & df["_image_available"]
        & df["_member_present"]
    )

    invalid = df.loc[~df["_row_valid"]].copy()
    invalid_path = args.out_dir / "temporal_excluded_rows.csv"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    invalid.to_csv(invalid_path, index=False, encoding="utf-8-sig")

    clean = df.loc[df["_row_valid"]].copy()
    clean = clean.drop_duplicates(subset=["candidate_id"], keep="first")
    clean = clean.sort_values(
        ["date_lower_bound", "pmcid", "image_ref_norm", "source_row_index"],
        kind="mergesort",
    ).reset_index(drop=True)

    if args.eval_limit is not None:
        clean = clean.iloc[: args.eval_limit].copy()
    if clean.empty:
        raise RuntimeError("No valid temporal rows remained after quality filtering.")

    duplicate_target = (
        clean.groupby("image_ref_norm")["image_archive_member"].nunique().gt(1)
    )
    if duplicate_target.any():
        conflicts = duplicate_target[duplicate_target].index.tolist()[:20]
        raise RuntimeError(
            "The same extracted image filename maps to multiple archive members: "
            f"{conflicts}"
        )

    image_dir = args.out_dir / "temporal_images"
    image_dir.mkdir(parents=True, exist_ok=True)
    unique_images = (
        clean[["image_ref_norm", "image_archive_member"]]
        .drop_duplicates("image_ref_norm")
        .reset_index(drop=True)
    )

    archive_sha = sha256_file(args.image_archive)
    with zipfile.ZipFile(args.image_archive, "r") as archive:
        member_lookup = {
            normalize_archive_member(name): name
            for name in archive.namelist()
            if not name.endswith("/")
        }
        requested_members = [
            normalize_archive_member(member)
            for member in unique_images["image_archive_member"].astype(str)
        ]
        missing_members = [
            member for member in requested_members
            if member not in member_lookup
        ]
        if missing_members:
            raise FileNotFoundError(
                f"{len(missing_members)} required image members are absent from the archive. "
                f"Examples: {missing_members[:10]}"
            )

        extracted = 0
        reused = 0
        for i, row in unique_images.iterrows():
            target = image_dir / str(row["image_ref_norm"])
            requested_member = normalize_archive_member(row["image_archive_member"])
            member = member_lookup[requested_member]
            if target.is_file() and target.stat().st_size > 0:
                reused += 1
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                temporary = target.with_suffix(target.suffix + ".partial")
                with archive.open(member, "r") as src, temporary.open("wb") as dst:
                    shutil.copyfileobj(src, dst, length=1024 * 1024)
                temporary.replace(target)
                extracted += 1
            if (i + 1) == 1 or (i + 1) % 250 == 0:
                log(
                    f"image preparation {i+1}/{len(unique_images)} "
                    f"extracted={extracted} reused={reused}"
                )

    corrupt: List[str] = []
    for image_name in unique_images["image_ref_norm"].astype(str):
        path = image_dir / image_name
        try:
            with Image.open(path) as image:
                image.verify()
        except Exception:
            corrupt.append(image_name)
    if corrupt:
        raise RuntimeError(
            f"Pillow verification failed for {len(corrupt)} extracted images: {corrupt[:20]}"
        )

    clean_csv = args.out_dir / "temporal_eval_clean.csv"
    clean_rows: List[Dict[str, Any]] = []
    for _, row in clean.iterrows():
        clean_rows.append({
            "candidate_id": row["candidate_id"],
            "pmcid": row["pmcid"],
            "date_lower_bound": row["date_lower_bound"],
            "date_upper_bound": row["date_upper_bound"],
            "source_file": row["source_file"],
            "source_row_index": int(row["source_row_index"]),
            "Figure_path": row["image_ref_norm"],
            "Question": row["Question"],
            "Choice A": row["Choice A"],
            "Choice B": row["Choice B"],
            "Choice C": row["Choice C"],
            "Choice D": row["Choice D"],
            "Answer": row["Answer"],
            "Answer_label": row["_label"],
        })
    write_csv(clean_csv, clean_rows)

    prep_summary = {
        "source_manifest": str(source_path),
        "source_manifest_sha256": sha256_file(source_path),
        "audit_cutoff": audit_summary.get("snapshot_cutoff"),
        "audit_gate": audit_summary.get("e4b_gate"),
        "source_rows": int(len(df)),
        "valid_rows_before_limit": int(df["_row_valid"].sum()),
        "excluded_rows": int((~df["_row_valid"]).sum()),
        "evaluated_rows": int(len(clean)),
        "unique_articles": int(clean["pmcid"].nunique()),
        "unique_images": int(clean["image_ref_norm"].nunique()),
        "label_distribution": {
            label: int((clean["_label"] == label).sum()) for label in VALID_LABELS
        },
        "date_lower_min": str(clean["date_lower_bound"].min()),
        "date_lower_max": str(clean["date_lower_bound"].max()),
        "image_archive": str(args.image_archive),
        "image_archive_sha256": archive_sha,
        "image_dir": str(image_dir),
        "clean_csv": str(clean_csv),
        "clean_csv_sha256": sha256_file(clean_csv),
        "smoke_limit": args.eval_limit,
    }
    write_json(args.out_dir / "temporal_preparation_summary.json", prep_summary)
    return clean_csv, image_dir, clean, prep_summary


def expected_feature_meta(
    train_mod: Any,
    model_name: str,
    csv_path: Path,
    row_count: int,
    context_length: int,
) -> Dict[str, Any]:
    if model_name == "biomed":
        model_id = train_mod.BIOMED_MODEL_ID
        score_protocol = train_mod.BIOMED_SCORE_PROTOCOL
    elif model_name == "clip":
        model_id = train_mod.CLIP_MODEL_ID
        score_protocol = train_mod.CLIP_SCORE_PROTOCOL
    else:
        raise ValueError(model_name)
    return {
        "model_id": model_id,
        "prompt_template": train_mod.PROMPT_TEMPLATE,
        "score_protocol": score_protocol,
        "csv_path": str(csv_path.resolve()),
        "csv_sha256": sha256_file(csv_path),
        "rows": int(row_count),
        "context_length": int(context_length),
        "resumable_cache_version": 1,
    }


def load_valid_final_cache(path: Path, meta: Mapping[str, Any]) -> Dict[str, Any] | None:
    if not path.is_file():
        return None
    payload = safe_torch_load(path)
    if payload.get("meta") != dict(meta):
        log(f"ignoring stale cache: {path}")
        return None
    if int(payload["gold_indices"].numel()) != int(meta["rows"]):
        log(f"ignoring incomplete cache: {path}")
        return None
    return payload


def chunk_ranges(total: int, chunk_size: int) -> Iterable[Tuple[int, int]]:
    for start in range(0, total, chunk_size):
        yield start, min(total, start + chunk_size)


def combine_chunks(
    chunk_dir: Path,
    ranges: Sequence[Tuple[int, int]],
    meta: Mapping[str, Any],
    final_path: Path,
) -> Dict[str, Any]:
    pair: List[torch.Tensor] = []
    scores: List[torch.Tensor] = []
    gold: List[torch.Tensor] = []
    rows: List[Mapping[str, Any]] = []
    for start, end in ranges:
        path = chunk_dir / f"chunk_{start:05d}_{end:05d}.pt"
        payload = safe_torch_load(path)
        if payload.get("meta") != dict(meta):
            raise RuntimeError(f"Stale or incompatible feature chunk: {path}")
        if int(payload.get("start", -1)) != start or int(payload.get("end", -1)) != end:
            raise RuntimeError(f"Feature chunk boundaries are inconsistent: {path}")
        pair.append(payload["pair_features"].float())
        scores.append(payload["base_scores"].float())
        gold.append(payload["gold_indices"].long())
        rows.extend(payload["rows"])
    output = {
        "meta": dict(meta),
        "pair_features": torch.cat(pair, dim=0).contiguous(),
        "base_scores": torch.cat(scores, dim=0).contiguous(),
        "gold_indices": torch.cat(gold, dim=0).contiguous(),
        "rows": list(rows),
    }
    if int(output["gold_indices"].numel()) != int(meta["rows"]):
        raise RuntimeError("Combined feature cache has an unexpected row count.")
    torch.save(output, final_path)
    return output


def extract_biomed_resumable(
    *,
    train_mod: Any,
    parsed: Sequence[Mapping[str, Any]],
    csv_path: Path,
    out_dir: Path,
    model: Any,
    preprocess: Any,
    tokenizer: Any,
    device: torch.device,
    context_length: int,
    chunk_size: int,
    progress_every: int,
    rebuild: bool,
) -> Dict[str, Any]:
    final_path = out_dir / "temporal_biomed_features.pt"
    meta = expected_feature_meta(train_mod, "biomed", csv_path, len(parsed), context_length)
    if rebuild:
        final_path.unlink(missing_ok=True)
        shutil.rmtree(out_dir / "biomed_chunks", ignore_errors=True)
    cached = load_valid_final_cache(final_path, meta)
    if cached is not None:
        log(f"loading complete BiomedCLIP cache: {final_path}")
        return cached

    chunk_dir = out_dir / "biomed_chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    ranges = list(chunk_ranges(len(parsed), chunk_size))
    started = time.time()
    use_amp = device.type == "cuda"
    completed_rows = 0

    for start, end in ranges:
        chunk_path = chunk_dir / f"chunk_{start:05d}_{end:05d}.pt"
        if chunk_path.is_file():
            old = safe_torch_load(chunk_path)
            if (
                old.get("meta") == meta
                and int(old.get("start", -1)) == start
                and int(old.get("end", -1)) == end
            ):
                completed_rows = end
                continue
            chunk_path.unlink()

        pair_features: List[torch.Tensor] = []
        base_scores: List[torch.Tensor] = []
        gold_indices: List[int] = []
        chunk_rows = parsed[start:end]
        for local_index, row in enumerate(chunk_rows, start=start):
            with Image.open(row["image_path"]) as raw:
                image_tensor = preprocess(raw.convert("RGB")).unsqueeze(0).to(device)
            prompts = train_mod.make_prompts(row["question"], row["options"])
            tokens = tokenizer(prompts, context_length=context_length)
            tokens = train_mod.move_tokens(tokens, device)
            with torch.inference_mode(), torch.autocast(
                device_type=device.type,
                dtype=torch.float16 if use_amp else torch.float32,
                enabled=use_amp,
            ):
                image_features = model.encode_image(image_tensor)
                image_features = image_features / image_features.norm(
                    dim=-1, keepdim=True
                ).clamp_min(1e-12)
                text_features = train_mod.biomed_encode_text(model, tokens)
                scores = (image_features @ text_features.T).squeeze(0)
            pair = image_features.float().expand_as(text_features.float()) * text_features.float()
            pair_features.append(pair.cpu())
            base_scores.append(scores.float().cpu())
            gold_indices.append(int(row["gold_index"]))
            done = local_index + 1
            if done == 1 or done % progress_every == 0:
                log(
                    f"BiomedCLIP processed={done}/{len(parsed)} "
                    f"elapsed_s={time.time()-started:.1f}"
                )

        payload = {
            "meta": meta,
            "start": start,
            "end": end,
            "pair_features": torch.stack(pair_features).contiguous(),
            "base_scores": torch.stack(base_scores).contiguous(),
            "gold_indices": torch.tensor(gold_indices, dtype=torch.long),
            "rows": train_mod.metadata_rows(chunk_rows),
        }
        torch.save(payload, chunk_path)
        completed_rows = end
        log(f"saved BiomedCLIP chunk rows={start}:{end}")

    if completed_rows != len(parsed):
        raise RuntimeError("BiomedCLIP extraction did not complete all rows.")
    return combine_chunks(chunk_dir, ranges, meta, final_path)


def extract_clip_resumable(
    *,
    train_mod: Any,
    parsed: Sequence[Mapping[str, Any]],
    csv_path: Path,
    out_dir: Path,
    model: Any,
    image_processor: Any,
    tokenizer: Any,
    device: torch.device,
    context_length: int,
    chunk_size: int,
    progress_every: int,
    rebuild: bool,
) -> Dict[str, Any]:
    final_path = out_dir / "temporal_clip_features.pt"
    meta = expected_feature_meta(train_mod, "clip", csv_path, len(parsed), context_length)
    if rebuild:
        final_path.unlink(missing_ok=True)
        shutil.rmtree(out_dir / "clip_chunks", ignore_errors=True)
    cached = load_valid_final_cache(final_path, meta)
    if cached is not None:
        log(f"loading complete CLIP cache: {final_path}")
        return cached

    chunk_dir = out_dir / "clip_chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    ranges = list(chunk_ranges(len(parsed), chunk_size))
    started = time.time()
    use_amp = device.type == "cuda"
    completed_rows = 0

    for start, end in ranges:
        chunk_path = chunk_dir / f"chunk_{start:05d}_{end:05d}.pt"
        if chunk_path.is_file():
            old = safe_torch_load(chunk_path)
            if (
                old.get("meta") == meta
                and int(old.get("start", -1)) == start
                and int(old.get("end", -1)) == end
            ):
                completed_rows = end
                continue
            chunk_path.unlink()

        pair_features: List[torch.Tensor] = []
        base_scores: List[torch.Tensor] = []
        gold_indices: List[int] = []
        chunk_rows = parsed[start:end]
        for local_index, row in enumerate(chunk_rows, start=start):
            with Image.open(row["image_path"]) as raw:
                image_inputs = image_processor(images=raw.convert("RGB"), return_tensors="pt")
            image_inputs = train_mod.move_mapping(image_inputs, device)
            prompts = train_mod.make_prompts(row["question"], row["options"])
            text_inputs = tokenizer(
                prompts,
                padding=True,
                truncation=True,
                max_length=context_length,
                return_tensors="pt",
            )
            text_inputs = train_mod.move_mapping(text_inputs, device)
            with torch.inference_mode(), torch.autocast(
                device_type=device.type,
                dtype=torch.float16 if use_amp else torch.float32,
                enabled=use_amp,
            ):
                image_features = train_mod.normalize_clip_output(
                    model.get_image_features(**image_inputs)
                )
                text_features = train_mod.normalize_clip_output(
                    model.get_text_features(**text_inputs)
                )
                scores = (image_features @ text_features.T).squeeze(0)
            pair = image_features.float().expand_as(text_features.float()) * text_features.float()
            pair_features.append(pair.cpu())
            base_scores.append(scores.float().cpu())
            gold_indices.append(int(row["gold_index"]))
            done = local_index + 1
            if done == 1 or done % progress_every == 0:
                log(
                    f"CLIP processed={done}/{len(parsed)} "
                    f"elapsed_s={time.time()-started:.1f}"
                )

        payload = {
            "meta": meta,
            "start": start,
            "end": end,
            "pair_features": torch.stack(pair_features).contiguous(),
            "base_scores": torch.stack(base_scores).contiguous(),
            "gold_indices": torch.tensor(gold_indices, dtype=torch.long),
            "rows": train_mod.metadata_rows(chunk_rows),
        }
        torch.save(payload, chunk_path)
        completed_rows = end
        log(f"saved CLIP chunk rows={start}:{end}")

    if completed_rows != len(parsed):
        raise RuntimeError("CLIP extraction did not complete all rows.")
    return combine_chunks(chunk_dir, ranges, meta, final_path)


def option_zscores(scores: torch.Tensor) -> torch.Tensor:
    mean = scores.mean(dim=1, keepdim=True)
    std = scores.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-6)
    return (scores - mean) / std


def top2_margin(scores: torch.Tensor) -> torch.Tensor:
    top2 = torch.topk(scores, k=2, dim=1).values
    return top2[:, 0] - top2[:, 1]


class LinearResidual(nn.Module):
    def __init__(self, input_dim: int, temperature: float = 20.0) -> None:
        super().__init__()
        self.temperature = float(temperature)
        self.net = nn.Linear(input_dim, 1)

    def forward(self, features: torch.Tensor, base: torch.Tensor) -> torch.Tensor:
        return self.temperature * (base + self.net(features).squeeze(-1))


def build_linear_features(
    biomed: Mapping[str, Any],
    clip: Mapping[str, Any],
    train_biomed: Mapping[str, Any],
    train_clip: Mapping[str, Any],
) -> torch.Tensor:
    train_bp = train_biomed["pair_features"].float()
    train_cp = train_clip["pair_features"].float()
    train_bs = train_biomed["base_scores"].float()
    train_cs = train_clip["base_scores"].float()

    bp_mean = train_bp.reshape(-1, train_bp.shape[-1]).mean(dim=0)
    bp_std = train_bp.reshape(-1, train_bp.shape[-1]).std(
        dim=0, unbiased=False
    ).clamp_min(1e-6)
    cp_mean = train_cp.reshape(-1, train_cp.shape[-1]).mean(dim=0)
    cp_std = train_cp.reshape(-1, train_cp.shape[-1]).std(
        dim=0, unbiased=False
    ).clamp_min(1e-6)
    bm = top2_margin(train_bs)
    cm = top2_margin(train_cs)
    bm_mean, bm_std = bm.mean(), bm.std(unbiased=False).clamp_min(1e-6)
    cm_mean, cm_std = cm.mean(), cm.std(unbiased=False).clamp_min(1e-6)

    bp = biomed["pair_features"].float()
    cp = clip["pair_features"].float()
    bs = biomed["base_scores"].float()
    cs = clip["base_scores"].float()
    bm_eval = ((top2_margin(bs) - bm_mean) / bm_std).view(-1, 1, 1).expand(-1, 4, 1)
    cm_eval = ((top2_margin(cs) - cm_mean) / cm_std).view(-1, 1, 1).expand(-1, 4, 1)
    return torch.cat(
        [
            (bp - bp_mean) / bp_std,
            (cp - cp_mean) / cp_std,
            option_zscores(bs).unsqueeze(-1),
            option_zscores(cs).unsqueeze(-1),
            bm_eval,
            cm_eval,
        ],
        dim=-1,
    ).contiguous()


@torch.inference_mode()
def linear_ensemble_logits(
    state_file: Path,
    features: torch.Tensor,
    base_scores: torch.Tensor,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    payload = safe_torch_load(state_file)
    spec = payload.get("spec", {})
    if spec.get("name") != "E1_linear" or not bool(spec.get("linear")):
        raise RuntimeError(f"Unexpected linear-state specification: {spec}")
    states = payload.get("state_dicts", [])
    if len(states) != 3:
        raise RuntimeError(f"Expected three linear seed states, found {len(states)}")

    seed_logits: List[torch.Tensor] = []
    for state in states:
        model = LinearResidual(features.shape[-1], temperature=20.0).to(device)
        model.load_state_dict(state)
        model.eval()
        pieces: List[torch.Tensor] = []
        for start in range(0, len(features), batch_size):
            end = min(len(features), start + batch_size)
            pieces.append(
                model(
                    features[start:end].to(device),
                    base_scores[start:end].to(device),
                ).cpu()
            )
        seed_logits.append(torch.cat(pieces, dim=0))
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return torch.stack(seed_logits, dim=0).mean(dim=0)


def exact_mcnemar_p(fixed: int, broken: int) -> float:
    n = fixed + broken
    if n == 0:
        return 1.0
    k = min(fixed, broken)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2.0 * tail)


def comparison_metrics(
    reference_correct: np.ndarray,
    candidate_correct: np.ndarray,
) -> Dict[str, Any]:
    reference_correct = reference_correct.astype(bool)
    candidate_correct = candidate_correct.astype(bool)
    fixed = int(np.sum(~reference_correct & candidate_correct))
    broken = int(np.sum(reference_correct & ~candidate_correct))
    n = int(reference_correct.size)
    return {
        "reference_correct": int(reference_correct.sum()),
        "reference_accuracy": float(reference_correct.mean()),
        "candidate_correct": int(candidate_correct.sum()),
        "candidate_accuracy": float(candidate_correct.mean()),
        "gain_correct": int(candidate_correct.sum() - reference_correct.sum()),
        "accuracy_delta": float(candidate_correct.mean() - reference_correct.mean()),
        "both_correct": int(np.sum(reference_correct & candidate_correct)),
        "both_wrong": int(np.sum(~reference_correct & ~candidate_correct)),
        "fixed": fixed,
        "broken": broken,
        "mcnemar_exact_p": exact_mcnemar_p(fixed, broken),
    }


def cluster_robust_difference(
    reference_correct: np.ndarray,
    candidate_correct: np.ndarray,
    clusters: Sequence[str],
) -> Dict[str, Any]:
    frame = pd.DataFrame({
        "cluster": pd.Series(clusters, dtype="string"),
        "n": 1,
        "d": candidate_correct.astype(float) - reference_correct.astype(float),
    })
    grouped = frame.groupby("cluster", sort=False).sum(numeric_only=True)
    counts = grouped["n"].to_numpy(dtype=float)
    deltas = grouped["d"].to_numpy(dtype=float)
    groups = int(len(grouped))
    rows = int(counts.sum())
    estimate = float(deltas.sum() / rows)
    residual = deltas - counts * estimate
    variance = float((groups / (groups - 1)) * np.sum(residual ** 2) / (rows ** 2))
    se = math.sqrt(max(variance, 0.0))
    normal = NormalDist()
    critical = float(normal.inv_cdf(0.975))
    statistic = estimate / se if se > 0 else 0.0
    p_value = float(2.0 * (1.0 - normal.cdf(abs(statistic)))) if se > 0 else 1.0
    return {
        "delta": estimate,
        "cluster_robust_se": se,
        "ci95": [estimate - critical * se, estimate + critical * se],
        "p_value": p_value,
        "clusters": groups,
        "rows": rows,
    }


def cluster_bootstrap(
    correctness: Mapping[str, np.ndarray],
    clusters: Sequence[str],
    repeats: int,
    seed: int,
) -> Dict[str, Any]:
    frame = pd.DataFrame({"cluster": pd.Series(clusters, dtype="string"), "n": 1})
    names = list(correctness)
    for name in names:
        frame[name] = correctness[name].astype(float)
    grouped = frame.groupby("cluster", sort=False).sum(numeric_only=True)
    counts = grouped["n"].to_numpy(dtype=np.int64)
    correct = grouped[names].to_numpy(dtype=np.int64)
    groups = len(grouped)
    rng = np.random.default_rng(seed)
    samples = {name: np.empty(repeats, dtype=np.float64) for name in names}
    batch = 100
    offset = 0
    while offset < repeats:
        width = min(batch, repeats - offset)
        indices = rng.integers(0, groups, size=(width, groups))
        denominators = counts[indices].sum(axis=1)
        numerators = correct[indices].sum(axis=1)
        values = numerators / denominators[:, None]
        for j, name in enumerate(names):
            samples[name][offset: offset + width] = values[:, j]
        offset += width

    output: Dict[str, Any] = {"accuracy_ci95": {}, "paired_delta_ci95": {}}
    for name in names:
        output["accuracy_ci95"][name] = [
            float(np.quantile(samples[name], 0.025)),
            float(np.quantile(samples[name], 0.975)),
        ]
    for reference, candidate in [
        ("biomedclip", "full"),
        ("biomedclip", "linear"),
        ("clip", "full"),
        ("full", "linear"),
    ]:
        delta = samples[candidate] - samples[reference]
        output["paired_delta_ci95"][f"{candidate}_minus_{reference}"] = [
            float(np.quantile(delta, 0.025)),
            float(np.quantile(delta, 0.975)),
        ]
    output["repeats"] = int(repeats)
    output["seed"] = int(seed)
    output["clusters"] = int(groups)
    return output


def label_distribution(predictions: np.ndarray) -> Dict[str, Any]:
    counts = Counter(chr(ord("A") + int(index)) for index in predictions.tolist())
    n = len(predictions)
    return {
        label: {"count": int(counts.get(label, 0)), "proportion": counts.get(label, 0) / n}
        for label in VALID_LABELS
    }


def linear_permutation_audit(
    state_file: Path,
    features_builder: Any,
    biomed: Mapping[str, Any],
    clip: Mapping[str, Any],
    train_biomed: Mapping[str, Any],
    train_clip: Mapping[str, Any],
    device: torch.device,
    batch_size: int,
) -> Dict[str, Any]:
    features = features_builder(biomed, clip, train_biomed, train_clip)
    original = linear_ensemble_logits(
        state_file, features, biomed["base_scores"].float(), device, batch_size
    ).argmax(dim=1)
    generator = torch.Generator().manual_seed(20260728)
    permutations = torch.stack([
        torch.randperm(4, generator=generator) for _ in range(len(original))
    ])
    row_indices = torch.arange(len(original)).unsqueeze(1)
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
    perm_features = features_builder(
        permuted_biomed, permuted_clip, train_biomed, train_clip
    )
    perm_pred = linear_ensemble_logits(
        state_file,
        perm_features,
        permuted_biomed["base_scores"].float(),
        device,
        batch_size,
    ).argmax(dim=1)
    mapped = permutations[torch.arange(len(original)), perm_pred]
    mismatches = int(mapped.ne(original).sum().item())
    return {
        "rows": int(len(original)),
        "permutation_seed": 20260728,
        "prediction_mismatches_after_unpermute": mismatches,
        "passed": mismatches == 0,
    }


def main() -> int:
    args = parse_args()
    resolve_defaults(args)
    audit_summary = validate_inputs(args)

    final_summary = args.out_dir / FINAL_SUMMARY_NAME
    if final_summary.exists() and args.eval_limit is None and not args.allow_overwrite:
        raise RuntimeError(
            f"Completed one-shot temporal result already exists: {final_summary}. "
            "Refusing to overwrite it."
        )

    clean_csv, image_dir, clean_df, prep_summary = prepare_temporal_dataset(
        args, audit_summary
    )
    log(
        f"prepared rows={len(clean_df)} articles={clean_df['pmcid'].nunique()} "
        f"images={clean_df['image_ref_norm'].nunique()}"
    )
    if args.prepare_only:
        log("prepare-only mode completed; no model was evaluated.")
        return 0

    train_mod = load_training_module()
    image_roots = [image_dir.resolve()]
    image_index = train_mod.index_images(image_roots)
    parsed = train_mod.parse_rows(clean_csv, len(clean_df), image_roots, image_index)
    if len(parsed) != len(clean_df):
        raise RuntimeError("Parsed temporal row count differs from the clean manifest.")

    approved = safe_torch_load(args.approved_manifest)
    required = {
        "state_paths", "validation_gate_pass", "holdout_gate_pass",
        "holdout_permutation_audit", "biomed_model_id", "clip_model_id",
        "prompt_template", "biomed_score_protocol", "clip_score_protocol",
    }
    missing = sorted(required.difference(approved))
    if missing:
        raise RuntimeError(f"Approved nonlinear manifest lacks fields: {missing}")
    if not bool(approved["validation_gate_pass"]) or not bool(approved["holdout_gate_pass"]):
        raise RuntimeError("The nonlinear ensemble was not approved by development/holdout gates.")
    if not bool(approved["holdout_permutation_audit"].get("passed", False)):
        raise RuntimeError("The approved nonlinear ensemble failed its prior permutation audit.")
    if approved["biomed_model_id"] != train_mod.BIOMED_MODEL_ID:
        raise RuntimeError("BiomedCLIP model identifier mismatch.")
    if approved["clip_model_id"] != train_mod.CLIP_MODEL_ID:
        raise RuntimeError("CLIP model identifier mismatch.")
    state_paths = [Path(path) for path in approved["state_paths"]]
    missing_states = [str(path) for path in state_paths if not path.is_file()]
    if missing_states:
        raise FileNotFoundError(f"Frozen nonlinear seed states are missing: {missing_states}")

    device = choose_device(args.device)
    log(f"device={device}")
    if device.type == "cuda":
        log(f"cuda_device={torch.cuda.get_device_name(0)}")

    # Load/extract BiomedCLIP features.
    biomed_cache_meta = expected_feature_meta(
        train_mod, "biomed", clean_csv, len(parsed), args.context_length
    )
    biomed_final = args.out_dir / "temporal_biomed_features.pt"
    biomed = None if args.rebuild_cache else load_valid_final_cache(
        biomed_final, biomed_cache_meta
    )
    if biomed is None:
        try:
            from open_clip import create_model_from_pretrained, get_tokenizer
        except ImportError as exc:
            raise RuntimeError("open_clip is not installed.") from exc
        log(f"loading BiomedCLIP={train_mod.BIOMED_MODEL_ID}")
        biomed_model, biomed_preprocess = create_model_from_pretrained(
            train_mod.BIOMED_MODEL_ID
        )
        biomed_tokenizer = get_tokenizer(train_mod.BIOMED_MODEL_ID)
        biomed_model = biomed_model.to(device).eval()
        for parameter in biomed_model.parameters():
            parameter.requires_grad_(False)
        biomed = extract_biomed_resumable(
            train_mod=train_mod,
            parsed=parsed,
            csv_path=clean_csv,
            out_dir=args.out_dir,
            model=biomed_model,
            preprocess=biomed_preprocess,
            tokenizer=biomed_tokenizer,
            device=device,
            context_length=args.context_length,
            chunk_size=args.chunk_size,
            progress_every=args.progress_every,
            rebuild=args.rebuild_cache,
        )
        del biomed_model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    else:
        log(f"loading complete BiomedCLIP cache: {biomed_final}")

    # Load/extract general CLIP features.
    try:
        from transformers import AutoImageProcessor, AutoTokenizer, CLIPModel
    except ImportError as exc:
        raise RuntimeError("transformers is not installed.") from exc
    clip_context = 77
    clip_cache_meta = expected_feature_meta(
        train_mod, "clip", clean_csv, len(parsed), clip_context
    )
    clip_final = args.out_dir / "temporal_clip_features.pt"
    clip = None if args.rebuild_cache else load_valid_final_cache(
        clip_final, clip_cache_meta
    )
    if clip is None:
        log(f"loading CLIP={train_mod.CLIP_MODEL_ID}")
        clip_model = CLIPModel.from_pretrained(
            train_mod.CLIP_MODEL_ID,
            local_files_only=args.local_files_only,
        ).to(device).eval()
        clip_tokenizer = AutoTokenizer.from_pretrained(
            train_mod.CLIP_MODEL_ID,
            local_files_only=args.local_files_only,
        )
        clip_processor = AutoImageProcessor.from_pretrained(
            train_mod.CLIP_MODEL_ID,
            local_files_only=args.local_files_only,
        )
        for parameter in clip_model.parameters():
            parameter.requires_grad_(False)
        clip_context = int(
            getattr(clip_model.config.text_config, "max_position_embeddings", 77)
        )
        clip = extract_clip_resumable(
            train_mod=train_mod,
            parsed=parsed,
            csv_path=clean_csv,
            out_dir=args.out_dir,
            model=clip_model,
            image_processor=clip_processor,
            tokenizer=clip_tokenizer,
            device=device,
            context_length=clip_context,
            chunk_size=args.chunk_size,
            progress_every=args.progress_every,
            rebuild=args.rebuild_cache,
        )
        del clip_model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    else:
        log(f"loading complete CLIP cache: {clip_final}")

    train_mod.assert_aligned(biomed, clip, "temporal")
    gold_t = biomed["gold_indices"].long()
    biomed_scores = biomed["base_scores"].float()
    clip_scores = clip["base_scores"].float()
    biomed_pred_t = biomed_scores.argmax(dim=1)
    clip_pred_t = clip_scores.argmax(dim=1)

    log("evaluating prespecified nonlinear ensemble")
    full_logits = train_mod.ensemble_logits(
        state_paths,
        biomed=biomed,
        clip=clip,
        device=device,
        batch_size=args.batch_size,
    )
    full_pred_t = full_logits.argmax(dim=1)
    full_permutation = train_mod.permutation_equivariance_audit(
        state_paths,
        biomed=biomed,
        clip=clip,
        device=device,
        batch_size=args.batch_size,
    )
    if not full_permutation["passed"]:
        raise RuntimeError("Nonlinear temporal permutation audit failed.")

    log("evaluating post-hoc linear ablation")
    train_biomed = safe_torch_load(args.train_feature_root / "train_biomed_features.pt")
    train_clip = safe_torch_load(args.train_feature_root / "train_clip_features.pt")
    train_mod.assert_aligned(train_biomed, train_clip, "linear training cache")
    linear_features = build_linear_features(biomed, clip, train_biomed, train_clip)
    linear_logits = linear_ensemble_logits(
        args.linear_states,
        linear_features,
        biomed_scores,
        device,
        args.batch_size,
    )
    linear_pred_t = linear_logits.argmax(dim=1)
    linear_permutation = linear_permutation_audit(
        args.linear_states,
        build_linear_features,
        biomed,
        clip,
        train_biomed,
        train_clip,
        device,
        args.batch_size,
    )
    if not linear_permutation["passed"]:
        raise RuntimeError("Linear temporal permutation audit failed.")

    gold = gold_t.numpy()
    predictions = {
        "biomedclip": biomed_pred_t.numpy(),
        "clip": clip_pred_t.numpy(),
        "full": full_pred_t.numpy(),
        "linear": linear_pred_t.numpy(),
    }
    correctness = {name: pred == gold for name, pred in predictions.items()}
    pmcids = clean_df["pmcid"].astype(str).tolist()

    comparisons: Dict[str, Any] = {}
    for reference, candidate in [
        ("biomedclip", "full"),
        ("biomedclip", "linear"),
        ("clip", "full"),
        ("full", "linear"),
    ]:
        key = f"{candidate}_vs_{reference}"
        metrics = comparison_metrics(correctness[reference], correctness[candidate])
        metrics["pmcid_cluster_robust"] = cluster_robust_difference(
            correctness[reference], correctness[candidate], pmcids
        )
        comparisons[key] = metrics

    bootstrap = cluster_bootstrap(
        correctness,
        pmcids,
        repeats=args.bootstrap_repeats,
        seed=args.bootstrap_seed,
    )

    prediction_rows: List[Dict[str, Any]] = []
    for i, row in clean_df.reset_index(drop=True).iterrows():
        record: Dict[str, Any] = {
            "candidate_id": row["candidate_id"],
            "pmcid": row["pmcid"],
            "date_lower_bound": row["date_lower_bound"],
            "date_upper_bound": row["date_upper_bound"],
            "image_ref": row["image_ref_norm"],
            "question": row["Question"],
            "choice_A": normalize_choice_text(row["Choice A"], "A"),
            "choice_B": normalize_choice_text(row["Choice B"], "B"),
            "choice_C": normalize_choice_text(row["Choice C"], "C"),
            "choice_D": normalize_choice_text(row["Choice D"], "D"),
            "gold_label": chr(ord("A") + int(gold[i])),
        }
        for name in predictions:
            record[f"{name}_pred_label"] = chr(ord("A") + int(predictions[name][i]))
            record[f"{name}_correct"] = int(correctness[name][i])
        if not correctness["biomedclip"][i] and correctness["full"][i]:
            record["full_transition_vs_biomedclip"] = "fixed"
        elif correctness["biomedclip"][i] and not correctness["full"][i]:
            record["full_transition_vs_biomedclip"] = "broken"
        elif correctness["biomedclip"][i] and correctness["full"][i]:
            record["full_transition_vs_biomedclip"] = "both_correct"
        else:
            record["full_transition_vs_biomedclip"] = "both_wrong"
        record["biomedclip_scores_json"] = json.dumps(biomed_scores[i].tolist())
        record["clip_scores_json"] = json.dumps(clip_scores[i].tolist())
        record["full_logits_json"] = json.dumps(full_logits[i].tolist())
        record["linear_logits_json"] = json.dumps(linear_logits[i].tolist())
        prediction_rows.append(record)
    predictions_path = args.out_dir / "temporal_predictions.csv"
    write_csv(predictions_path, prediction_rows)

    model_results = {
        name: {
            "correct": int(values.sum()),
            "accuracy": float(values.mean()),
            "pmcid_cluster_bootstrap_ci95": bootstrap["accuracy_ci95"][name],
            "prediction_label_distribution": label_distribution(predictions[name]),
        }
        for name, values in correctness.items()
    }

    status = "final_one_shot_temporal_evaluation" if args.eval_limit is None else "smoke_test"
    summary: Dict[str, Any] = {
        "experiment": "paper_e4b_post_biomedclip_cutoff_temporal_evaluation_v1",
        "status": status,
        "primary_model": "full",
        "primary_reference": "biomedclip",
        "secondary_model": "linear_posthoc_ablation",
        "no_training_or_selection_on_temporal_set": True,
        "preparation": prep_summary,
        "evaluated_rows": int(len(clean_df)),
        "unique_articles": int(clean_df["pmcid"].nunique()),
        "unique_images": int(clean_df["image_ref_norm"].nunique()),
        "model_ids": {
            "biomedclip": train_mod.BIOMED_MODEL_ID,
            "clip": train_mod.CLIP_MODEL_ID,
        },
        "prompt_template": train_mod.PROMPT_TEMPLATE,
        "approved_nonlinear_manifest": str(args.approved_manifest),
        "approved_nonlinear_manifest_sha256": sha256_file(args.approved_manifest),
        "linear_states": str(args.linear_states),
        "linear_states_sha256": sha256_file(args.linear_states),
        "model_results": model_results,
        "comparisons": comparisons,
        "cluster_bootstrap": bootstrap,
        "permutation_audits": {
            "full": full_permutation,
            "linear": linear_permutation,
        },
        "output_predictions": str(predictions_path),
        "output_predictions_sha256": sha256_file(predictions_path),
        "environment": {
            "torch_version": torch.__version__,
            "device": str(device),
            "cuda_device": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        },
        "limitations": [
            "Publication date is a conservative proxy for exclusion from the June 2022 PMC-15M snapshot; the exact pretraining manifest is unavailable.",
            "PMC-VQA questions are automatically generated and retain source-dataset quality limitations.",
            "The nonlinear model is the prespecified primary method; the linear head was designed after earlier ablation analysis and is secondary/exploratory.",
            "Questions are clustered within articles; article-clustered uncertainty is reported alongside row-level McNemar tests.",
        ],
    }
    summary_path = args.out_dir / (
        FINAL_SUMMARY_NAME if args.eval_limit is None else "temporal_smoke_summary.json"
    )
    write_json(summary_path, summary)

    report_lines = [
        "E4B POST-CUTOFF TEMPORAL EVALUATION",
        "=" * 78,
        f"Status: {status}",
        f"Rows: {len(clean_df)}",
        f"Articles: {clean_df['pmcid'].nunique()}",
        f"Images: {clean_df['image_ref_norm'].nunique()}",
        "",
        "MODEL RESULTS",
    ]
    for name in ["biomedclip", "clip", "full", "linear"]:
        result = model_results[name]
        report_lines.append(
            f"{name}: {result['correct']}/{len(clean_df)} = {result['accuracy']:.6f}; "
            f"article-bootstrap CI95={result['pmcid_cluster_bootstrap_ci95']}"
        )
    report_lines.extend(["", "PRIMARY COMPARISON: FULL VS BIOMEDCLIP"])
    primary = comparisons["full_vs_biomedclip"]
    report_lines.extend([
        f"gain_correct={primary['gain_correct']}",
        f"accuracy_delta={primary['accuracy_delta']:.6f}",
        f"fixed/broken={primary['fixed']}/{primary['broken']}",
        f"row_mcnemar_exact_p={primary['mcnemar_exact_p']:.8g}",
        f"pmcid_cluster_robust={json.dumps(primary['pmcid_cluster_robust'])}",
        f"pmcid_cluster_bootstrap_delta_CI95={bootstrap['paired_delta_ci95']['full_minus_biomedclip']}",
        "",
        "PERMUTATION AUDITS",
        f"full={json.dumps(full_permutation)}",
        f"linear={json.dumps(linear_permutation)}",
        "",
        f"Outputs: {args.out_dir}",
    ])
    (args.out_dir / "REPORT.txt").write_text(
        "\n".join(report_lines) + "\n", encoding="utf-8"
    )

    log(
        f"completed full={model_results['full']['accuracy']:.4f} "
        f"biomedclip={model_results['biomedclip']['accuracy']:.4f} "
        f"linear={model_results['linear']['accuracy']:.4f} "
        f"clip={model_results['clip']['accuracy']:.4f}"
    )
    log(f"outputs={args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
