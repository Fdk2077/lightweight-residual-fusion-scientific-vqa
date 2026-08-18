#!/usr/bin/env python3
"""Audit PMC-VQA for a publication-date-disjoint temporal evaluation set.

Purpose
-------
This script identifies PMC-VQA rows whose source articles were published
*definitely after* the BiomedCLIP/PMC-15M snapshot cutoff (default:
2022-06-15), while excluding every article and image already used by the
paper's train, development, holdout, or canonical-test splits.

What it establishes
-------------------
- A row is a temporal candidate only when NCBI PMC metadata gives a lower
  publication-date bound strictly later than the cutoff.
- It does not claim that every pre-cutoff article was used for BiomedCLIP
  pretraining; the exact PMC-15M manifest is unavailable.

Expected project layout (defaults)
----------------------------------
D:/thesis-vqa/
  data/raw/pmc_vqa/train.csv
  data/raw/pmc_vqa/test.csv
  data/raw/pmc_vqa/test_clean.csv       (optional; deduplicated)
  outputs/joint_clip_biomed/...

The audit is resumable: NCBI ESummary records are cached after every batch.
No model checkpoint is evaluated in this stage.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import pandas as pd

try:
    import torch
except Exception as exc:  # pragma: no cover - environment-specific
    torch = None
    TORCH_IMPORT_ERROR = exc
else:
    TORCH_IMPORT_ERROR = None


DEFAULT_CUTOFF = dt.date(2022, 6, 15)
VALID_LABELS = ("A", "B", "C", "D")
PMCID_RE = re.compile(r"(?<![A-Z0-9])PMC(\d+)", re.IGNORECASE)
SPACE_RE = re.compile(r"\s+")
CHOICE_PREFIX_TEMPLATE = r"^\\s*{label}\\s*[:.)]\\s*"

REQUIRED_COLUMNS = (
    "Figure_path",
    "Question",
    "Choice A",
    "Choice B",
    "Choice C",
    "Choice D",
    "Answer_label",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Find definitely post-cutoff PMC-VQA articles and build a "
            "leakage-audited temporal-candidate manifest."
        )
    )
    p.add_argument(
        "--project-root",
        type=Path,
        default=Path("D:/thesis-vqa"),
    )
    p.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Default: <project-root>/data/raw/pmc_vqa",
    )
    p.add_argument(
        "--joint-root",
        type=Path,
        default=None,
        help="Default: <project-root>/outputs/joint_clip_biomed",
    )
    p.add_argument(
        "--source-csv",
        action="append",
        type=Path,
        default=None,
        help=(
            "Repeatable. Defaults to train.csv, test.csv, and test_clean.csv "
            "when present. The *_2.csv variant is intentionally excluded."
        ),
    )
    p.add_argument(
        "--image-root",
        action="append",
        type=Path,
        default=None,
        help=(
            "Repeatable image directory. If omitted, common raw/processed "
            "locations are scanned when they exist."
        ),
    )
    p.add_argument(
        "--image-archive",
        action="append",
        type=Path,
        default=None,
        help=(
            "Repeatable ZIP archive containing PMC-VQA images. If omitted, "
            "images.zip and images_2.zip under --data-dir are checked when present."
        ),
    )
    p.add_argument(
        "--seed-cache",
        action="append",
        type=Path,
        default=None,
        help=(
            "Optional existing NCBI ESummary cache(s). The previous overlap "
            "audit cache is used automatically when found."
        ),
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help=(
            "Default: <project-root>/outputs/paper_e4a_temporal_audit_v1"
        ),
    )
    p.add_argument("--cutoff", default=DEFAULT_CUTOFF.isoformat())
    p.add_argument("--batch-size", type=int, default=150)
    p.add_argument("--request-delay", type=float, default=0.40)
    p.add_argument("--retries", type=int, default=5)
    p.add_argument("--email", default=os.environ.get("NCBI_EMAIL"))
    p.add_argument("--api-key", default=os.environ.get("NCBI_API_KEY"))
    p.add_argument(
        "--offline",
        action="store_true",
        help="Use cached NCBI records only; do not perform network requests.",
    )
    p.add_argument(
        "--max-new-fetch",
        type=int,
        default=0,
        help=(
            "Safety limit for new PMCIDs fetched in this run; 0 means all. "
            "Useful for a smoke run such as --max-new-fetch 300."
        ),
    )
    p.add_argument("--min-rows", type=int, default=300)
    p.add_argument("--min-articles", type=int, default=150)
    p.add_argument("--min-images", type=int, default=250)
    p.add_argument(
        "--require-image-files",
        action="store_true",
        help=(
            "Require at least --min-images available image assets, either "
            "as extracted files or inside ZIP archives, for a PASS decision. "
            "Without this flag, image availability is "
            "reported but does not block metadata qualification."
        ),
    )
    return p.parse_args()


def log(message: str) -> None:
    print(f"[E4A] {message}", flush=True)


def scalar_to_text(value: Any) -> str:
    """Convert a scalar value to text without evaluating pandas.NA as bool."""
    if value is None or value is pd.NA:
        return ""
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        missing = False
    if isinstance(missing, bool) and missing:
        return ""
    return str(value)


def normalize_text(value: Any) -> str:
    text = scalar_to_text(value).strip().lower()
    return SPACE_RE.sub(" ", text)


def normalize_choice_text(value: Any, expected_label: str) -> str:
    """Remove only the expected option-label prefix, preserving text like B-mode or d)."""
    raw = scalar_to_text(value).strip()
    pattern = CHOICE_PREFIX_TEMPLATE.format(label=re.escape(expected_label))
    raw = re.sub(pattern, "", raw, count=1, flags=re.IGNORECASE)
    return normalize_text(raw)


def normalize_image_ref(value: Any) -> str:
    text = scalar_to_text(value).strip().replace("\\", "/")
    return Path(text).name.lower()


def extract_pmcid(value: Any) -> str:
    match = PMCID_RE.search(scalar_to_text(value))
    return f"PMC{match.group(1)}" if match else ""


def read_csv_robust(path: Path, **kwargs: Any) -> pd.DataFrame:
    last_error: Optional[Exception] = None
    for encoding in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return pd.read_csv(
                path,
                encoding=encoding,
                low_memory=False,
                **kwargs,
            )
        except Exception as exc:  # noqa: BLE001
            last_error = exc
    raise RuntimeError(f"Unable to read {path}: {last_error}")


def resolve_source_csvs(args: argparse.Namespace) -> List[Path]:
    if args.source_csv:
        paths = [p.resolve() for p in args.source_csv]
    else:
        candidates = [
            args.data_dir / "train.csv",
            args.data_dir / "test.csv",
            args.data_dir / "test_clean.csv",
        ]
        paths = [p.resolve() for p in candidates if p.is_file()]
    if not paths:
        raise FileNotFoundError(
            f"No source CSV found under {args.data_dir}. Expected train.csv "
            "and/or test.csv."
        )
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    return paths


def load_and_clean_sources(paths: Sequence[Path]) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    frames: List[pd.DataFrame] = []
    reports: List[Dict[str, Any]] = []

    for path in paths:
        df = read_csv_robust(path)
        df.columns = [str(c).strip() for c in df.columns]
        missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"{path} is missing required columns: {missing}")

        original_rows = len(df)
        keep = list(REQUIRED_COLUMNS)
        if "Answer" in df.columns:
            keep.append("Answer")
        df = df[keep].copy()
        df["source_file"] = path.name
        df["source_row_index"] = np_arange_safe(len(df))

        for col in REQUIRED_COLUMNS:
            df[col] = df[col].astype("string").str.strip()
        if "Answer" in df.columns:
            df["Answer"] = df["Answer"].astype("string").str.strip()
        else:
            df["Answer"] = pd.Series([pd.NA] * len(df), dtype="string")

        df["Answer_label"] = df["Answer_label"].str.upper()
        missing_required = df[list(REQUIRED_COLUMNS)].isna().any(axis=1)
        empty_required = pd.Series(False, index=df.index)
        for col in REQUIRED_COLUMNS:
            empty_required |= df[col].fillna("").str.len().eq(0)
        invalid_label = ~df["Answer_label"].isin(VALID_LABELS)

        removed_missing = int((missing_required | empty_required).sum())
        removed_invalid = int((~(missing_required | empty_required) & invalid_label).sum())
        df = df.loc[~(missing_required | empty_required | invalid_label)].copy()

        df["image_ref_norm"] = df["Figure_path"].map(normalize_image_ref)
        df["pmcid"] = df["Figure_path"].map(extract_pmcid)
        missing_pmcid = int(df["pmcid"].eq("").sum())

        choice_norm = pd.DataFrame({
            label: df[f"Choice {label}"].map(
                lambda value, expected=label: normalize_choice_text(value, expected)
            )
            for label in VALID_LABELS
        })
        df["choices_unique"] = choice_norm.nunique(axis=1).eq(4)
        df["gold_choice"] = [
            str(row[f"Choice {row['Answer_label']}"])
            for _, row in df.iterrows()
        ]
        answer_norm = pd.Series([
            normalize_choice_text(answer, label)
            for answer, label in zip(df["Answer"], df["Answer_label"])
        ], index=df.index)
        gold_norm = pd.Series([
            normalize_choice_text(choice, label)
            for choice, label in zip(df["gold_choice"], df["Answer_label"])
        ], index=df.index)
        answer_as_label = df["Answer"].map(normalize_text).str.upper()
        df["answer_matches_gold_choice"] = (
            (answer_norm.ne("") & answer_norm.eq(gold_norm))
            | answer_as_label.eq(df["Answer_label"])
        )

        reports.append({
            "source_file": path.name,
            "original_rows": int(original_rows),
            "clean_rows": int(len(df)),
            "removed_missing_required": removed_missing,
            "removed_invalid_label": removed_invalid,
            "rows_without_pmcid_after_cleaning": missing_pmcid,
        })
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    before = len(combined)
    dedup_cols = [
        "Figure_path", "Question", "Choice A", "Choice B", "Choice C",
        "Choice D", "Answer_label"
    ]
    combined = combined.drop_duplicates(subset=dedup_cols, keep="first").reset_index(drop=True)
    reports.append({
        "source_file": "<combined>",
        "original_rows": int(before),
        "clean_rows": int(len(combined)),
        "removed_cross_file_duplicates": int(before - len(combined)),
    })
    return combined, reports


def np_arange_safe(n: int) -> List[int]:
    # Avoid a hard NumPy dependency; pandas accepts a plain list efficiently.
    return list(range(n))


def load_pt_rows(path: Path) -> List[Mapping[str, Any]]:
    if torch is None:
        raise RuntimeError(
            "PyTorch is required to read feature caches but could not be "
            f"imported: {TORCH_IMPORT_ERROR}"
        )
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or not isinstance(payload.get("rows"), list):
        raise ValueError(f"Feature cache has no list-valued 'rows': {path}")
    return payload["rows"]


def collect_used_identities(joint_root: Path) -> Tuple[set[str], set[str], List[Dict[str, Any]]]:
    used_pmcids: set[str] = set()
    used_images: set[str] = set()
    reports: List[Dict[str, Any]] = []

    feature_paths = [
        joint_root / "joint_train12000_dev1000_multiseed_v1" / "train_biomed_features.pt",
        joint_root / "joint_train12000_dev1000_multiseed_v1" / "val_biomed_features.pt",
    ]
    csv_paths = [
        joint_root / "joint_holdout290_ensemble_v1" / "holdout_predictions.csv",
        joint_root / "joint_canonical_test2000_ensemble_v1" / "test_predictions.csv",
    ]

    for path in feature_paths:
        if not path.is_file():
            reports.append({"path": str(path), "status": "missing"})
            continue
        rows = load_pt_rows(path)
        local_pmcids = set()
        local_images = set()
        for row in rows:
            image_ref = row.get("image_ref", row.get("image_path", ""))
            pmcid = extract_pmcid(image_ref)
            image_norm = normalize_image_ref(image_ref)
            if pmcid:
                local_pmcids.add(pmcid)
            if image_norm:
                local_images.add(image_norm)
        used_pmcids.update(local_pmcids)
        used_images.update(local_images)
        reports.append({
            "path": str(path),
            "status": "loaded",
            "rows": len(rows),
            "unique_pmcids": len(local_pmcids),
            "unique_images": len(local_images),
        })

    for path in csv_paths:
        if not path.is_file():
            reports.append({"path": str(path), "status": "missing"})
            continue
        df = read_csv_robust(path)
        image_col = next(
            (c for c in ("image_ref", "Figure_path", "image_path") if c in df.columns),
            None,
        )
        if image_col is None:
            reports.append({"path": str(path), "status": "no_image_column"})
            continue
        image_values = df[image_col].fillna("").astype(str)
        local_pmcids = set(filter(None, image_values.map(extract_pmcid)))
        local_images = set(filter(None, image_values.map(normalize_image_ref)))
        used_pmcids.update(local_pmcids)
        used_images.update(local_images)
        reports.append({
            "path": str(path),
            "status": "loaded",
            "rows": int(len(df)),
            "unique_pmcids": len(local_pmcids),
            "unique_images": len(local_images),
        })

    if not used_pmcids or not used_images:
        raise RuntimeError(
            "Could not build a complete used-split exclusion set. Ensure the "
            "joint train/dev feature caches and holdout/test prediction files "
            f"exist under {joint_root}. Details: {reports}"
        )
    return used_pmcids, used_images, reports


def chunks(items: Sequence[str], size: int) -> Iterable[List[str]]:
    for start in range(0, len(items), size):
        yield list(items[start:start + size])


def request_json(url: str, retries: int) -> Dict[str, Any]:
    last_error: Optional[Exception] = None
    for attempt in range(1, retries + 1):
        try:
            request = urllib.request.Request(
                url,
                headers={
                    "User-Agent": (
                        "thesis-vqa-e4a-temporal-audit/1.0 "
                        "(NCBI E-utilities metadata audit)"
                    )
                },
            )
            with urllib.request.urlopen(request, timeout=90) as response:
                return json.loads(response.read().decode("utf-8"))
        except (
            urllib.error.URLError,
            urllib.error.HTTPError,
            TimeoutError,
            json.JSONDecodeError,
        ) as exc:
            last_error = exc
            if attempt == retries:
                break
            sleep_seconds = min(30.0, 2.0 ** attempt)
            log(
                f"NCBI request failed attempt={attempt}/{retries}: {exc}; "
                f"sleeping {sleep_seconds:.1f}s"
            )
            time.sleep(sleep_seconds)
    raise RuntimeError(f"NCBI request failed after {retries} attempts: {last_error}")


def fetch_esummary_batch(
    pmcids: Sequence[str],
    email: Optional[str],
    api_key: Optional[str],
    retries: int,
) -> Dict[str, Dict[str, Any]]:
    numeric_ids = [pmcid[3:] for pmcid in pmcids]
    params: Dict[str, str] = {
        "db": "pmc",
        "id": ",".join(numeric_ids),
        "retmode": "json",
        "tool": "thesis_vqa_e4a_temporal_audit",
    }
    if email:
        params["email"] = email
    if api_key:
        params["api_key"] = api_key
    url = (
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?"
        + urllib.parse.urlencode(params)
    )
    payload = request_json(url, retries=retries)
    result = payload.get("result", {})
    output: Dict[str, Dict[str, Any]] = {}
    for pmcid, numeric_id in zip(pmcids, numeric_ids):
        record = result.get(numeric_id)
        output[pmcid] = (
            record if isinstance(record, dict)
            else {"_missing": True, "_numeric_id": numeric_id}
        )
    return output


def parse_sort_pub_date(value: Any) -> Optional[dt.date]:
    text = scalar_to_text(value).strip()
    match = re.search(r"\b(\d{4})[/-](\d{1,2})[/-](\d{1,2})\b", text)
    if not match:
        return None
    year, month, day = map(int, match.groups())
    try:
        return dt.date(year, month, day)
    except ValueError:
        return None


MONTHS = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "sept": 9, "oct": 10,
    "nov": 11, "dec": 12,
}


def parse_textual_date_bounds(value: Any) -> Optional[Tuple[dt.date, dt.date, str]]:
    text = scalar_to_text(value).strip().lower()
    if not text:
        return None
    exact = parse_sort_pub_date(text)
    if exact:
        return exact, exact, "day"
    year_match = re.search(r"\b(?:19|20)\d{2}\b", text)
    if not year_match:
        return None
    year = int(year_match.group(0))
    month: Optional[int] = None
    for token, number in MONTHS.items():
        if re.search(rf"\b{re.escape(token)}\b", text):
            month = number
            break
    tail = text[year_match.end():]
    day_match = re.search(r"\b([0-3]?\d)\b", tail)
    day = int(day_match.group(1)) if day_match else None
    if month is None:
        return dt.date(year, 1, 1), dt.date(year, 12, 31), "year"
    if day is not None:
        try:
            exact = dt.date(year, month, day)
            return exact, exact, "day"
        except ValueError:
            pass
    next_month = (
        dt.date(year + 1, 1, 1)
        if month == 12 else dt.date(year, month + 1, 1)
    )
    return dt.date(year, month, 1), next_month - dt.timedelta(days=1), "month"


def determine_date_bounds(record: Mapping[str, Any]) -> Tuple[Optional[dt.date], Optional[dt.date], str, str]:
    candidates = (
        ("sortpubdate", record.get("sortpubdate")),
        ("epubdate", record.get("epubdate")),
        ("pubdate", record.get("pubdate")),
    )
    for source, value in candidates:
        exact = parse_sort_pub_date(value)
        if exact:
            return exact, exact, "day", source
    parsed: List[Tuple[dt.date, dt.date, str, str]] = []
    for source, value in candidates[1:]:
        bounds = parse_textual_date_bounds(value)
        if bounds:
            parsed.append((*bounds, source))
    if not parsed:
        return None, None, "missing", "none"
    # Conservative earliest plausible bounds across available publication fields.
    low = min(x[0] for x in parsed)
    high = min(x[1] for x in parsed)
    selected = min(parsed, key=lambda x: x[0])
    return low, high, selected[2], selected[3]


def classify_date(low: Optional[dt.date], high: Optional[dt.date], cutoff: dt.date) -> str:
    if low is None or high is None:
        return "date_missing"
    if low > cutoff:
        return "definitely_post_snapshot_by_publication_date"
    if high <= cutoff:
        return "potentially_pretraining_eligible_by_publication_date"
    return "date_interval_crosses_cutoff"


def merge_seed_caches(paths: Sequence[Path], destination: Path) -> Dict[str, Dict[str, Any]]:
    merged: Dict[str, Dict[str, Any]] = {}
    if destination.is_file():
        try:
            payload = json.loads(destination.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                merged.update(payload)
        except Exception as exc:  # noqa: BLE001
            log(f"warning: could not read destination cache {destination}: {exc}")
    for path in paths:
        if not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                merged.update(payload)
                log(f"seeded {len(payload)} NCBI records from {path}")
        except Exception as exc:  # noqa: BLE001
            log(f"warning: could not read seed cache {path}: {exc}")
    destination.write_text(
        json.dumps(merged, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return merged


def resolve_image_roots(args: argparse.Namespace) -> List[Path]:
    if args.image_root:
        roots = [p.resolve() for p in args.image_root]
    else:
        candidates = [
            args.data_dir / "images",
            args.data_dir / "images_2",
            args.project_root / "data/processed/pmc_vqa_v1/images",
        ]
        roots = [p.resolve() for p in candidates if p.is_dir()]
    return roots


def resolve_image_archives(args: argparse.Namespace) -> List[Path]:
    if args.image_archive:
        archives = [p.resolve() for p in args.image_archive]
    else:
        candidates = [args.data_dir / "images.zip", args.data_dir / "images_2.zip"]
        archives = [p.resolve() for p in candidates if p.is_file()]
    return archives


def index_available_images(
    roots: Sequence[Path],
    archives: Sequence[Path],
    wanted: set[str],
) -> Tuple[Dict[str, str], Dict[str, str]]:
    if not wanted:
        return {}, {}
    found_files: Dict[str, str] = {}
    found_archives: Dict[str, str] = {}
    remaining = set(wanted)
    for root in roots:
        log(f"indexing candidate images under {root}")
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            name = path.name.lower()
            if name in remaining:
                found_files[name] = str(path.resolve())
                remaining.remove(name)
                if not remaining:
                    return found_files, found_archives
    for archive in archives:
        if not remaining:
            break
        log(f"indexing candidate image names in archive {archive}")
        try:
            with zipfile.ZipFile(archive, "r") as zf:
                for info in zf.infolist():
                    if info.is_dir():
                        continue
                    name = Path(info.filename).name.lower()
                    if name in remaining:
                        found_archives[name] = f"{archive.resolve()}::{info.filename}"
                        remaining.remove(name)
                        if not remaining:
                            break
        except (zipfile.BadZipFile, OSError) as exc:
            log(f"warning: could not index image archive {archive}: {exc}")
    return found_files, found_archives


def write_csv(path: Path, df: pd.DataFrame) -> None:
    df.to_csv(path, index=False, encoding="utf-8-sig")


def label_distribution(series: pd.Series) -> Dict[str, Dict[str, float | int]]:
    counts = series.value_counts().reindex(VALID_LABELS, fill_value=0)
    total = int(counts.sum())
    return {
        label: {
            "count": int(counts[label]),
            "proportion": float(counts[label] / total) if total else 0.0,
        }
        for label in VALID_LABELS
    }


def stable_row_id(row: Mapping[str, Any]) -> str:
    key = "\x1f".join([
        str(row.get("Figure_path", "")),
        str(row.get("Question", "")),
        str(row.get("Choice A", "")),
        str(row.get("Choice B", "")),
        str(row.get("Choice C", "")),
        str(row.get("Choice D", "")),
        str(row.get("Answer_label", "")),
    ])
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:20]


def main() -> int:
    args = parse_args()
    args.project_root = args.project_root.resolve()
    args.data_dir = (args.data_dir or args.project_root / "data/raw/pmc_vqa").resolve()
    args.joint_root = (
        args.joint_root or args.project_root / "outputs/joint_clip_biomed"
    ).resolve()
    args.out_dir = (
        args.out_dir
        or args.project_root / "outputs/paper_e4a_temporal_audit_v1"
    ).resolve()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    cutoff = dt.date.fromisoformat(args.cutoff)

    if args.batch_size <= 0 or args.batch_size > 200:
        raise ValueError("--batch-size must be between 1 and 200")
    if args.max_new_fetch < 0:
        raise ValueError("--max-new-fetch cannot be negative")

    source_paths = resolve_source_csvs(args)
    log("source CSVs: " + ", ".join(str(p) for p in source_paths))
    rows, source_reports = load_and_clean_sources(source_paths)
    log(f"clean deduplicated rows={len(rows):,}")

    used_pmcids, used_images, used_reports = collect_used_identities(args.joint_root)
    log(
        f"exclusion set: articles={len(used_pmcids):,} "
        f"images={len(used_images):,}"
    )

    rows["excluded_used_pmcid"] = rows["pmcid"].isin(used_pmcids)
    rows["excluded_used_image"] = rows["image_ref_norm"].isin(used_images)
    rows["excluded_missing_pmcid"] = rows["pmcid"].eq("")
    eligible = rows.loc[
        ~(
            rows["excluded_used_pmcid"]
            | rows["excluded_used_image"]
            | rows["excluded_missing_pmcid"]
        )
    ].copy()
    log(
        f"metadata rows after split exclusion={len(eligible):,}; "
        f"unique articles={eligible['pmcid'].nunique():,}; "
        f"unique images={eligible['image_ref_norm'].nunique():,}"
    )

    seed_caches: List[Path] = list(args.seed_cache or [])
    auto_seed = (
        args.project_root
        / "outputs/audits/biomedclip_pmc_overlap_clean2000_v2/ncbi_esummary_cache.json"
    )
    if auto_seed.is_file():
        seed_caches.append(auto_seed)
    cache_path = args.out_dir / "ncbi_esummary_cache.json"
    cache = merge_seed_caches([p.resolve() for p in seed_caches], cache_path)

    unique_pmcids = sorted(
        eligible["pmcid"].unique().tolist(),
        key=lambda x: int(x[3:]),
    )
    pending = [pmcid for pmcid in unique_pmcids if pmcid not in cache]
    if args.max_new_fetch:
        pending = pending[:args.max_new_fetch]
    log(
        f"NCBI metadata: cached among candidates={sum(p in cache for p in unique_pmcids):,}; "
        f"pending this run={len(pending):,}; total candidates={len(unique_pmcids):,}"
    )

    if pending and args.offline:
        log("offline mode: pending PMCIDs will remain unaudited")
    elif pending:
        for batch_number, batch in enumerate(chunks(pending, args.batch_size), start=1):
            log(
                f"fetch batch {batch_number}/{(len(pending) + args.batch_size - 1) // args.batch_size}: "
                f"n={len(batch)} first={batch[0]} last={batch[-1]}"
            )
            cache.update(
                fetch_esummary_batch(
                    batch,
                    email=args.email,
                    api_key=args.api_key,
                    retries=args.retries,
                )
            )
            cache_path.write_text(
                json.dumps(cache, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            if batch_number * args.batch_size < len(pending):
                time.sleep(args.request_delay)

    article_rows: List[Dict[str, Any]] = []
    for pmcid in unique_pmcids:
        record = cache.get(pmcid)
        if not isinstance(record, dict):
            article_rows.append({
                "pmcid": pmcid,
                "metadata_status": "not_fetched",
                "overlap_status": "date_missing",
                "snapshot_cutoff": cutoff.isoformat(),
            })
            continue
        low, high, granularity, date_source = determine_date_bounds(record)
        article_rows.append({
            "pmcid": pmcid,
            "numeric_pmc_id": pmcid[3:],
            "title": record.get("title", ""),
            "journal": record.get("fulljournalname", ""),
            "sortpubdate_raw": record.get("sortpubdate", ""),
            "epubdate_raw": record.get("epubdate", ""),
            "pubdate_raw": record.get("pubdate", ""),
            "date_lower_bound": low.isoformat() if low else "",
            "date_upper_bound": high.isoformat() if high else "",
            "date_granularity": granularity,
            "date_source": date_source,
            "snapshot_cutoff": cutoff.isoformat(),
            "overlap_status": classify_date(low, high, cutoff),
            "metadata_status": "missing_at_ncbi" if record.get("_missing") else "fetched",
        })
    articles = pd.DataFrame(article_rows)

    article_counts = eligible.groupby("pmcid", as_index=False).agg(
        candidate_row_count=("pmcid", "size"),
        candidate_image_count=("image_ref_norm", "nunique"),
    )
    articles = articles.merge(article_counts, on="pmcid", how="left")

    eligible = eligible.merge(
        articles[[
            "pmcid", "title", "journal", "date_lower_bound",
            "date_upper_bound", "date_granularity", "date_source",
            "overlap_status", "metadata_status"
        ]],
        on="pmcid",
        how="left",
    )
    temporal = eligible.loc[
        eligible["overlap_status"].eq(
            "definitely_post_snapshot_by_publication_date"
        )
    ].copy()
    temporal["candidate_id"] = [stable_row_id(row) for row in temporal.to_dict("records")]

    image_roots = resolve_image_roots(args)
    image_archives = resolve_image_archives(args)
    available_file_map, available_archive_map = index_available_images(
        image_roots,
        image_archives,
        set(temporal["image_ref_norm"].unique()),
    )
    available_names = set(available_file_map) | set(available_archive_map)
    temporal["image_available"] = temporal["image_ref_norm"].isin(available_names)
    temporal["resolved_image_path"] = temporal["image_ref_norm"].map(available_file_map).fillna("")
    temporal["image_archive_member"] = temporal["image_ref_norm"].map(available_archive_map).fillna("")

    # Explicit leakage assertions.
    pmcid_overlap = sorted(set(temporal["pmcid"]) & used_pmcids)
    image_overlap = sorted(set(temporal["image_ref_norm"]) & used_images)
    if pmcid_overlap or image_overlap:
        raise AssertionError(
            f"Temporal manifest leakage detected: pmcid={len(pmcid_overlap)}, "
            f"image={len(image_overlap)}"
        )

    # Stable ordering groups every article and image together.
    temporal = temporal.sort_values(
        ["date_lower_bound", "pmcid", "image_ref_norm", "source_file", "source_row_index"],
        kind="stable",
    ).reset_index(drop=True)

    status_counts = articles["overlap_status"].fillna("date_missing").value_counts().to_dict()
    row_status_counts = eligible["overlap_status"].fillna("date_missing").value_counts().to_dict()
    n_rows = int(len(temporal))
    n_articles = int(temporal["pmcid"].nunique())
    n_images = int(temporal["image_ref_norm"].nunique())
    n_available_images = int(
        temporal.loc[temporal["image_available"], "image_ref_norm"].nunique()
    )
    label_dist = label_distribution(temporal["Answer_label"])
    proportions = [label_dist[label]["proportion"] for label in VALID_LABELS]
    severe_label_skew = bool(proportions and (max(proportions) > 0.50 or min(proportions) < 0.10))

    pass_reasons: List[str] = []
    fail_reasons: List[str] = []
    checks = {
        "minimum_rows": n_rows >= args.min_rows,
        "minimum_articles": n_articles >= args.min_articles,
        "minimum_unique_images": n_images >= args.min_images,
        "zero_pmcid_overlap": len(pmcid_overlap) == 0,
        "zero_image_overlap": len(image_overlap) == 0,
        "no_severe_label_skew": not severe_label_skew,
    }
    if args.require_image_files:
        checks["minimum_available_image_files"] = n_available_images >= args.min_images
    for name, passed in checks.items():
        (pass_reasons if passed else fail_reasons).append(name)

    complete_metadata = all(p in cache for p in unique_pmcids)
    if not complete_metadata:
        fail_reasons.append("ncbi_metadata_incomplete")
    decision = "PASS_FOR_E4B" if not fail_reasons else "NOT_READY_FOR_E4B"

    selected_columns = [
        "candidate_id", "source_file", "source_row_index", "Figure_path",
        "image_ref_norm", "resolved_image_path", "image_archive_member",
        "image_available", "pmcid",
        "date_lower_bound", "date_upper_bound", "date_granularity",
        "date_source", "title", "journal", "Question", "Choice A", "Choice B",
        "Choice C", "Choice D", "Answer", "Answer_label", "gold_choice",
        "choices_unique", "answer_matches_gold_choice",
    ]
    write_csv(args.out_dir / "temporal_candidate_manifest_all.csv", temporal[selected_columns])
    write_csv(args.out_dir / "temporal_article_audit.csv", articles)
    write_csv(args.out_dir / "source_cleaning_report.csv", pd.DataFrame(source_reports))
    write_csv(args.out_dir / "used_split_inventory.csv", pd.DataFrame(used_reports))

    quality_flags = temporal.loc[
        (~temporal["choices_unique"])
        | (~temporal["answer_matches_gold_choice"])
        | (~temporal["image_available"])
    , selected_columns]
    write_csv(args.out_dir / "temporal_candidate_quality_flags.csv", quality_flags)

    summary = {
        "run_name": args.out_dir.name,
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "snapshot_cutoff": cutoff.isoformat(),
        "source_csvs": [str(p) for p in source_paths],
        "source_cleaning": source_reports,
        "used_split_exclusion": {
            "unique_pmcids": len(used_pmcids),
            "unique_images": len(used_images),
            "inventory": used_reports,
        },
        "after_split_exclusion": {
            "rows": int(len(eligible)),
            "unique_pmcids": int(eligible["pmcid"].nunique()),
            "unique_images": int(eligible["image_ref_norm"].nunique()),
        },
        "ncbi": {
            "cache_file": str(cache_path),
            "candidate_pmcids": len(unique_pmcids),
            "cached_or_fetched": sum(p in cache for p in unique_pmcids),
            "complete": complete_metadata,
            "offline": bool(args.offline),
            "max_new_fetch": int(args.max_new_fetch),
        },
        "article_status_counts": {str(k): int(v) for k, v in status_counts.items()},
        "row_status_counts": {str(k): int(v) for k, v in row_status_counts.items()},
        "temporal_candidates": {
            "rows": n_rows,
            "articles": n_articles,
            "unique_images": n_images,
            "available_unique_image_assets": n_available_images,
            "image_roots": [str(p) for p in image_roots],
            "image_archives": [str(p) for p in image_archives],
            "available_as_extracted_files": len(available_file_map),
            "available_inside_archives": len(available_archive_map),
            "label_distribution": label_dist,
            "rows_with_nonunique_choices": int((~temporal["choices_unique"]).sum()),
            "rows_where_answer_text_differs_from_gold_choice": int((~temporal["answer_matches_gold_choice"]).sum()),
            "pmcid_overlap_with_existing_splits": len(pmcid_overlap),
            "image_overlap_with_existing_splits": len(image_overlap),
        },
        "e4b_gate": {
            "decision": decision,
            "thresholds": {
                "min_rows": args.min_rows,
                "min_articles": args.min_articles,
                "min_images": args.min_images,
                "require_image_files": bool(args.require_image_files),
            },
            "checks": checks,
            "passed": pass_reasons,
            "failed": fail_reasons,
        },
        "interpretation": {
            "definitely_post_snapshot_by_publication_date": (
                "The conservative lower bound of the returned publication "
                "date is later than the 2022-06-15 snapshot cutoff."
            ),
            "potentially_pretraining_eligible_by_publication_date": (
                "Publication timing is on/before the cutoff; exact PMC-15M "
                "membership is not established."
            ),
            "date_interval_crosses_cutoff": (
                "Available date precision cannot place the article wholly "
                "before or after the cutoff."
            ),
        },
        "limitations": [
            "Publication date is not identical to PMC-OA ingestion date.",
            "The exact PMC-15M training manifest is unavailable.",
            "A post-cutoff publication set removes temporal eligibility for "
            "the cited snapshot but does not address all possible external "
            "pretraining sources.",
            "Rows are automatically generated PMC-VQA questions and retain "
            "the quality limitations of the source dataset.",
        ],
    }
    (args.out_dir / "temporal_audit_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    report_lines = [
        "E4A TEMPORAL CANDIDATE AUDIT",
        "=" * 78,
        f"Cutoff: {cutoff.isoformat()}",
        f"Clean source rows: {len(rows):,}",
        f"Rows after current-split exclusion: {len(eligible):,}",
        f"Candidate PMCIDs audited: {len(unique_pmcids):,}",
        f"NCBI metadata complete: {complete_metadata}",
        "",
        "DEFINITELY POST-CUTOFF CANDIDATES",
        f"Rows: {n_rows:,}",
        f"Articles: {n_articles:,}",
        f"Unique images: {n_images:,}",
        f"Available unique image assets: {n_available_images:,}",
        f"Label distribution: {json.dumps(label_dist, ensure_ascii=False)}",
        f"PMCID overlap with current splits: {len(pmcid_overlap)}",
        f"Image overlap with current splits: {len(image_overlap)}",
        "",
        f"E4B GATE: {decision}",
        f"Passed checks: {', '.join(pass_reasons) or 'none'}",
        f"Failed checks: {', '.join(fail_reasons) or 'none'}",
        "",
        "IMPORTANT",
        "Only the conservative NCBI publication-date lower bound is used to "
        "declare an article definitely post-cutoff. No model was evaluated "
        "during E4A.",
    ]
    (args.out_dir / "REPORT.txt").write_text("\n".join(report_lines), encoding="utf-8")
    print("\n".join(report_lines))
    print(f"\nOutputs: {args.out_dir}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        raise SystemExit(130)
