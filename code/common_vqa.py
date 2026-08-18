#!/usr/bin/env python3
"""Shared utilities for thesis VQA chapter-4 continuation experiments.

These utilities are intentionally conservative: they try to discover common
column names and project paths, but fail with clear messages when a required
field is missing.
"""
from __future__ import annotations

import csv
import json
import math
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    from PIL import Image, ImageFilter, ImageEnhance, ImageDraw
except Exception as exc:  # pragma: no cover
    raise RuntimeError("Pillow is required. Install with: pip install pillow") from exc


IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


class UserFacingError(RuntimeError):
    pass


def log(msg: str) -> None:
    print(f"[chapter4-next] {msg}", flush=True)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        raise UserFacingError(f"CSV not found: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = [dict(r) for r in reader]
    if not rows:
        raise UserFacingError(f"CSV is empty: {path}")
    return rows


def write_csv_rows(path: Path, rows: Sequence[Dict[str, Any]], fieldnames: Optional[Sequence[str]] = None) -> None:
    ensure_dir(path.parent)
    if fieldnames is None:
        keys: List[str] = []
        for row in rows:
            for key in row.keys():
                if key not in keys:
                    keys.append(key)
        fieldnames = keys

    def _write(target: Path) -> None:
        with target.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(fieldnames), extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

    try:
        _write(path)
    except PermissionError:
        # On Windows/Git Bash this usually means the previous CSV is still open
        # in Excel or another viewer. Do not abort the experiment; write a
        # timestamped fallback so the results are not lost.
        fallback = path.with_name(f"{path.stem}_{time.strftime('%Y%m%d_%H%M%S')}{path.suffix}")
        _write(fallback)
        log(f"WARNING: Could not overwrite {path}; wrote fallback file: {fallback}")


def write_json(path: Path, obj: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def find_first_existing(paths: Iterable[Path]) -> Optional[Path]:
    for p in paths:
        if p and p.exists():
            return p
    return None


def discover_pmc_csv(project_root: Path, data_root: Optional[Path], kind: str = "clean_test") -> Optional[Path]:
    """Find a likely PMC-VQA CSV file.

    kind: clean_test | train | val | validation | test
    """
    roots = [project_root]
    if data_root:
        roots.append(data_root)
    names_by_kind = {
        "clean_test": ["clean_test.csv", "csv.clean_test", "pmc_test_clean_2000.csv", "test_clean_2000.csv", "pmc_clean_test.csv"],
        "train": ["train.csv", "csv.train", "pmc_train.csv", "train_2000.csv", "pmc_train_2000.csv"],
        "validation": ["validation.csv", "val.csv", "csv.val", "pmc_val.csv", "validation_200.csv", "pmc_validation_200.csv"],
        "test": ["test.csv", "csv.test", "pmc_test.csv"],
    }
    wanted = names_by_kind.get(kind, names_by_kind["clean_test"])
    for root in roots:
        for name in wanted:
            direct = root / name
            if direct.exists():
                return direct
        # semantic search, but avoid huge hidden folders
        candidates: List[Path] = []
        for p in root.rglob("*"):
            if not p.is_file():
                continue
            low = p.name.lower()
            if low in {n.lower() for n in wanted}:
                candidates.append(p)
            elif kind == "clean_test" and "clean" in low and "test" in low and low.endswith((".csv", ".txt")):
                candidates.append(p)
            elif kind == "train" and "train" in low and low.endswith((".csv", ".txt")):
                candidates.append(p)
            elif kind == "validation" and ("val" in low or "validation" in low) and low.endswith((".csv", ".txt")):
                candidates.append(p)
        if candidates:
            candidates.sort(key=lambda p: (len(str(p)), str(p)))
            return candidates[0]
    return None


def _is_excluded_path(path: Path) -> bool:
    """Return True for folders that must not be scanned for dataset images."""
    excluded = {
        ".venv", "venv", "env", ".git", "__pycache__", "cache", "huggingface",
        "outputs", "reports", "scripts", "src", "temp", "tmp", "checkpoints",
    }
    parts = {part.lower() for part in path.parts}
    return bool(parts & excluded)


def _count_direct_images(path: Path) -> int:
    try:
        return sum(1 for child in path.iterdir() if child.is_file() and child.suffix.lower() in IMAGE_EXTS)
    except (PermissionError, OSError):
        return 0


def discover_image_roots(project_root: Path, data_root: Optional[Path]) -> List[Path]:
    """Find likely dataset image roots.

    v3 deliberately avoids scanning virtualenv/cache folders. The previous
    version could pick matplotlib package images or HuggingFace cache folders,
    which made all PMC rows appear as skipped when the actual dataset images
    were not prioritized.
    """
    roots: List[Path] = []
    bases = [project_root]
    if data_root:
        bases.append(data_root)

    priority_names = [
        "data/processed/pmc_vqa_v1/images",
        "data/processed/pmc_vqa/images",
        "data/raw/pmc_vqa/images",
        "pmc-vqa/images",
        "pmc_vqa/images",
        "PMC-VQA/images",
        "images",
        "Images",
        "processed_images",
        "pmc_images",
        "figures",
        "data/images",
        "data/processed/images",
    ]
    for base in bases:
        if not base or not base.exists():
            continue
        for name in priority_names:
            p = base / name
            if p.exists() and p.is_dir() and not _is_excluded_path(p):
                roots.append(p)

    # Controlled recursive scan for large image folders, excluding non-dataset dirs.
    for base in bases:
        if not base or not base.exists():
            continue
        for p in base.rglob("*"):
            if not p.is_dir() or _is_excluded_path(p):
                continue
            if _count_direct_images(p) >= 20:
                roots.append(p)

    # De-duplicate while preserving order. Prefer roots with more direct images.
    seen = set()
    out = []
    for r in roots:
        try:
            rp = str(r.resolve())
        except OSError:
            continue
        if rp not in seen:
            seen.add(rp)
            out.append(r)

    out.sort(key=lambda r: (-_count_direct_images(r), len(str(r)), str(r)))
    return out


def _image_keys(path: Path, root: Path) -> set[str]:
    keys = {path.name, path.stem, path.name.lower(), path.stem.lower()}
    try:
        rel = path.relative_to(root)
        rel_posix = rel.as_posix()
        rel_win = str(rel).replace("/", "\\")
        keys.update({
            rel_posix,
            rel_win,
            rel_posix.lower(),
            rel_win.lower(),
        })
    except ValueError:
        pass
    return keys


def index_images(image_roots: Sequence[Path], max_files: Optional[int] = None) -> Dict[str, Path]:
    index: Dict[str, Path] = {}
    n = 0
    for root in image_roots:
        if not root.exists():
            continue
        for p in root.rglob("*"):
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
                for k in _image_keys(p, root):
                    index.setdefault(k, p)
                n += 1
                if max_files and n >= max_files:
                    return index
    return index


def first_nonempty(row: Dict[str, Any], names: Sequence[str]) -> Optional[str]:
    lower = {str(k).lower(): k for k in row.keys()}
    for name in names:
        key = lower.get(name.lower())
        if key is not None:
            val = row.get(key)
            if val is not None and str(val).strip() != "":
                return str(val).strip()
    return None


def get_question(row: Dict[str, Any]) -> str:
    val = first_nonempty(row, ["question", "Question", "q", "prompt", "query"])
    if not val:
        raise UserFacingError(f"Could not find question column. Available columns: {list(row.keys())}")
    return val


def get_image_ref(row: Dict[str, Any]) -> str:
    val = first_nonempty(row, ["image", "image_path", "path", "img_path", "image_file", "filename", "file_name", "image_id", "img_id", "figure_path"])
    if not val:
        raise UserFacingError(f"Could not find image reference column. Available columns: {list(row.keys())}")
    return val


def strip_choice_prefix(value: Any, expected_label: Optional[str] = None) -> str:
    """Remove dataset option prefixes such as ``A:``, ``B)``, or ``C -``.

    Prefixes are metadata identifying option position and must not be included in
    the candidate text scored by BLIP. Keeping them can create a label-position
    bias when candidate loss is used for multiple-choice evaluation.
    """
    text = "" if value is None else str(value).strip()
    if expected_label:
        pattern = rf"^\s*{re.escape(expected_label)}\s*[:.)\-]\s*"
    else:
        pattern = r"^\s*[A-Ea-e]\s*[:.)\-]\s*"
    return re.sub(pattern, "", text, count=1, flags=re.IGNORECASE).strip()


def clean_options(options: Sequence[Any]) -> List[str]:
    """Return option texts with positional prefixes removed."""
    cleaned: List[str] = []
    for i, option in enumerate(options):
        label = chr(ord("A") + i) if i < 5 else None
        cleaned.append(strip_choice_prefix(option, label))
    return cleaned


def get_options(row: Dict[str, Any]) -> List[str]:
    # Common PMC-VQA formats: option_A/B/C/D, choice_a, A, B, C, D, answer_A...
    patterns = [
        ["option_A", "option_B", "option_C", "option_D"],
        ["option_a", "option_b", "option_c", "option_d"],
        ["choice_A", "choice_B", "choice_C", "choice_D"],
        ["choice_a", "choice_b", "choice_c", "choice_d"],
        ["Choice A", "Choice B", "Choice C", "Choice D"],
        ["choice a", "choice b", "choice c", "choice d"],
        ["Choice_A", "Choice_B", "Choice_C", "Choice_D"],
        ["ChoiceA", "ChoiceB", "ChoiceC", "ChoiceD"],
        ["choiceA", "choiceB", "choiceC", "choiceD"],
        ["A", "B", "C", "D"],
        ["a", "b", "c", "d"],
        ["answer_A", "answer_B", "answer_C", "answer_D"],
    ]
    lower_map = {str(k).lower(): k for k in row.keys()}
    for names in patterns:
        vals = []
        ok = True
        for name in names:
            key = lower_map.get(name.lower())
            if key is None or str(row.get(key, "")).strip() == "":
                ok = False
                break
            vals.append(str(row[key]).strip())
        if ok:
            return clean_options(vals)
    # Some datasets store options as JSON/list in one column.
    opt_blob = first_nonempty(row, ["options", "choices", "candidate_answers", "multiple_choices"])
    if opt_blob:
        try:
            parsed = json.loads(opt_blob)
            if isinstance(parsed, dict):
                vals = [parsed[k] for k in sorted(parsed.keys()) if str(k).lower()[:1] in "abcd"]
                if len(vals) >= 2:
                    return clean_options([str(v).strip() for v in vals])
            if isinstance(parsed, list) and len(parsed) >= 2:
                return clean_options([str(v).strip() for v in parsed])
        except Exception:
            # Try splitting A: ... B: ...
            parts = re.split(r"\s*(?:\||;|\n)\s*", opt_blob)
            parts = [re.sub(r"^[A-Da-d][\).:-]\s*", "", p).strip() for p in parts if p.strip()]
            if len(parts) >= 2:
                return clean_options(parts)
    return []


def get_answer_text(row: Dict[str, Any], options: Sequence[str]) -> str:
    # Prefer exact answer text, fall back to answer label/index.
    val = first_nonempty(row, ["answer", "answers", "correct_answer", "gt_answer", "label_text", "target", "gold", "ground_truth"])
    if val:
        # If answer is A/B/C/D or 0/1/2/3, map to option text.
        mapped = map_label_to_option(val, options)
        return mapped if mapped is not None else val
    label = first_nonempty(row, ["label", "answer_label", "Answer_label", "correct", "correct_option", "correct_label", "answer_idx", "answer_index", "gt", "target_idx"])
    if label:
        mapped = map_label_to_option(label, options)
        if mapped is not None:
            return mapped
        return label
    if options:
        raise UserFacingError(f"Could not find answer/label column. Available columns: {list(row.keys())}")
    return ""


def get_answer_label(row: Dict[str, Any], options: Sequence[str]) -> Optional[str]:
    val = first_nonempty(row, ["label", "answer_label", "Answer_label", "correct", "correct_option", "correct_label", "answer_idx", "answer_index", "gt", "target_idx"])
    if val:
        lab = normalize_label(val)
        if lab:
            return lab
    ans = first_nonempty(row, ["answer", "correct_answer", "gt_answer", "label_text", "target", "gold", "ground_truth"])
    if ans and options:
        mapped_lab = answer_text_to_label(ans, options)
        if mapped_lab:
            return mapped_lab
    return None


def normalize_label(label: str) -> Optional[str]:
    s = str(label).strip()
    if not s:
        return None
    # Remove common prefixes.
    s2 = re.sub(r"^(option|choice|answer)\s*", "", s, flags=re.I).strip()
    s2 = re.sub(r"[\).:-].*$", "", s2).strip()
    if s2.upper() in ["A", "B", "C", "D", "E"]:
        return s2.upper()
    if s2.isdigit():
        i = int(s2)
        if 0 <= i <= 4:
            # Accept both 0-based and 1-based; 0 maps A, 1 maps B if 0-based? Ambiguous.
            # Most CSV labels use 0-based when numeric. Use 0->A, 1->B, ...
            return chr(ord("A") + i)
        if 1 <= i <= 5:
            return chr(ord("A") + i - 1)
    return None


def map_label_to_option(label: str, options: Sequence[str]) -> Optional[str]:
    lab = normalize_label(label)
    if lab:
        idx = ord(lab) - ord("A")
        if 0 <= idx < len(options):
            return str(options[idx])
    return None


def answer_text_to_label(answer: str, options: Sequence[str]) -> Optional[str]:
    ans = normalize_text(answer)
    for i, opt in enumerate(options):
        if normalize_text(opt) == ans:
            return chr(ord("A") + i)
    return None


def resolve_image_path(ref: str, image_index: Dict[str, Path], roots: Sequence[Path]) -> Optional[Path]:
    if not ref:
        return None
    ref_path = Path(ref)
    if ref_path.exists():
        return ref_path
    for root in roots:
        p = root / ref
        if p.exists():
            return p
        if ref_path.name:
            p2 = root / ref_path.name
            if p2.exists():
                return p2
    key_candidates = [ref, Path(ref).name, Path(ref).stem]
    # Add common extensions if only id is given.
    if Path(ref).suffix == "":
        key_candidates.extend([ref + ext for ext in IMAGE_EXTS])
    for key in key_candidates:
        if key in image_index:
            return image_index[key]
    return None


def normalize_text(text: Any) -> str:
    s = str(text or "").strip().lower()
    s = re.sub(r"^[a-d][\).:-]\s*", "", s, flags=re.I)
    s = re.sub(r"[^\w\s\u0600-\u06FF.-]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def label_from_prediction(prediction: str, options: Sequence[str]) -> Optional[str]:
    s = str(prediction or "").strip()
    lab = normalize_label(s)
    if lab:
        return lab
    s_norm = normalize_text(s)
    # Exact text match or contains option text.
    for i, opt in enumerate(options):
        opt_norm = normalize_text(opt)
        if opt_norm and (s_norm == opt_norm or opt_norm in s_norm or s_norm in opt_norm):
            return chr(ord("A") + i)
    return None


def accuracy(rows: Sequence[Dict[str, Any]], pred_key: str = "pred_label", gold_key: str = "gold_label") -> float:
    total = 0
    ok = 0
    for r in rows:
        gold = r.get(gold_key)
        pred = r.get(pred_key)
        if gold in (None, ""):
            continue
        total += 1
        if str(pred).strip().upper() == str(gold).strip().upper():
            ok += 1
    return ok / total if total else math.nan


def bootstrap_ci_binary(correct: Sequence[bool], z: float = 1.96) -> Tuple[float, float, float]:
    n = len(correct)
    if n == 0:
        return math.nan, math.nan, math.nan
    p = sum(correct) / n
    # Wilson interval, no random dependency.
    denom = 1 + z*z/n
    centre = (p + z*z/(2*n)) / denom
    half = z * math.sqrt((p*(1-p) + z*z/(4*n)) / n) / denom
    return p, max(0.0, centre-half), min(1.0, centre+half)


def mcnemar_exact(b01: int, b10: int) -> float:
    """Two-sided exact McNemar p-value using binomial p=0.5."""
    n = b01 + b10
    if n == 0:
        return 1.0
    k = min(b01, b10)
    # p = 2 * sum_{i=0}^k C(n,i) 0.5^n
    prob = 0.0
    for i in range(k + 1):
        prob += math.comb(n, i) * (0.5 ** n)
    return min(1.0, 2 * prob)


def perturb_image(img: Image.Image, kind: str) -> Image.Image:
    img = img.convert("RGB")
    w, h = img.size
    if kind == "clean":
        return img
    if kind == "blur":
        return img.filter(ImageFilter.GaussianBlur(radius=2.0))
    if kind == "low_light":
        return ImageEnhance.Brightness(img).enhance(0.55)
    if kind == "low_resolution":
        small = img.resize((max(1, w // 3), max(1, h // 3)), Image.Resampling.BILINEAR)
        return small.resize((w, h), Image.Resampling.BILINEAR)
    if kind == "occlusion":
        out = img.copy()
        draw = ImageDraw.Draw(out)
        # Central semi-large rectangle: deterministic and simple.
        x0, y0 = int(w * 0.35), int(h * 0.35)
        x1, y1 = int(w * 0.65), int(h * 0.65)
        draw.rectangle([x0, y0, x1, y1], fill=(128, 128, 128))
        return out
    if kind == "compression":
        # JPEG roundtrip.
        import io
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=35)
        buf.seek(0)
        return Image.open(buf).convert("RGB")
    raise ValueError(f"Unknown perturbation kind: {kind}")


def format_float(x: Any, ndigits: int = 4) -> str:
    try:
        f = float(x)
    except Exception:
        return "ندارد" if x in (None, "") else str(x)
    if math.isnan(f):
        return "ندارد"
    return f"{f:.{ndigits}f}"


def latex_escape(s: Any) -> str:
    text = str(s)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(ch, ch) for ch in text)


def write_latex_table(path: Path, caption: str, label: str, headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> None:
    ensure_dir(path.parent)
    colspec = "|" + "|".join(["c"] * len(headers)) + "|"
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        rf"\caption{{{latex_escape(caption)}}}",
        rf"\label{{{label}}}",
        rf"\begin{{tabular}}{{{colspec}}}",
        r"\hline",
        " & ".join(latex_escape(h) for h in headers) + r" \\",
        r"\hline",
    ]
    for row in rows:
        lines.append(" & ".join(latex_escape(x) for x in row) + r" \\")
        lines.append(r"\hline")
    lines += [r"\end{tabular}", r"\end{table}", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def load_blip_model(model_name: str, lora_path: Optional[Path] = None, device: str = "cuda", local_files_only: bool = False):
    import torch
    from transformers import AutoProcessor, BlipForQuestionAnswering

    log(f"Loading processor/model: {model_name}")
    processor = AutoProcessor.from_pretrained(model_name, local_files_only=local_files_only)
    model = BlipForQuestionAnswering.from_pretrained(model_name, local_files_only=local_files_only)
    if lora_path:
        if not _is_valid_lora_adapter_dir(Path(lora_path)):
            raise UserFacingError(
                "Invalid LoRA adapter directory: "
                f"{lora_path}. Expected adapter_config.json plus adapter_model.safetensors/bin."
            )
        try:
            from peft import PeftModel
        except Exception as exc:
            raise UserFacingError("PEFT is required for LoRA evaluation. Install with: pip install peft") from exc
        log(f"Loading LoRA adapter: {lora_path}")
        try:
            model = PeftModel.from_pretrained(model, str(lora_path), local_files_only=local_files_only)
        except Exception as exc:
            raise UserFacingError(f"Could not load LoRA adapter from {lora_path}: {exc}") from exc
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    return processor, model, device


def score_options_blip(processor, model, device: str, image: Image.Image, question: str, options: Sequence[str]) -> Tuple[str, str, List[float]]:
    """Return predicted label/text and per-option losses. Lower loss is better.

    Positional prefixes (A:/B:/...) are removed before scoring to prevent the
    option label itself from affecting candidate loss.
    """
    import torch
    scored_options = clean_options(options)
    losses: List[float] = []
    with torch.no_grad():
        for opt in scored_options:
            inputs = processor(images=image, text=question, return_tensors="pt")
            labels = processor.tokenizer(str(opt), return_tensors="pt", padding=False).input_ids
            inputs = {k: v.to(device) for k, v in inputs.items()}
            labels = labels.to(device)
            out = model(**inputs, labels=labels)
            loss = float(out.loss.detach().cpu().item())
            losses.append(loss)
    best = min(range(len(losses)), key=lambda i: losses[i])
    return chr(ord("A") + best), str(scored_options[best]), losses


def generate_answer_blip(processor, model, device: str, image: Image.Image, question: str, max_new_tokens: int = 20) -> str:
    import torch
    with torch.no_grad():
        inputs = processor(images=image, text=question, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        out = model.generate(**inputs, max_new_tokens=max_new_tokens)
        return processor.decode(out[0], skip_special_tokens=True).strip()


def _is_valid_lora_adapter_dir(path: Path) -> bool:
    """Return True only for a real PEFT adapter directory.

    HuggingFace cache may contain negative-cache entries under `.no_exist` with
    empty placeholder files. Those paths must not be treated as local LoRA
    adapters.
    """
    config_path = path / "adapter_config.json"
    weight_candidates = [
        path / "adapter_model.safetensors",
        path / "adapter_model.bin",
        path / "pytorch_model.bin",
    ]
    if not config_path.is_file():
        return False
    if not any(w.is_file() and w.stat().st_size > 0 for w in weight_candidates):
        return False
    try:
        if config_path.stat().st_size <= 2:
            return False
        obj = json.loads(config_path.read_text(encoding="utf-8"))
        if not isinstance(obj, dict) or not obj:
            return False
        # PEFT LoRA adapters normally include peft_type=LORA or at least r/target_modules.
        peft_type = str(obj.get("peft_type", "")).upper()
        if peft_type and peft_type != "LORA":
            return False
        if peft_type == "LORA" or "r" in obj or "target_modules" in obj:
            return True
    except Exception:
        return False
    return False


def discover_lora_adapters(project_root: Path) -> List[Path]:
    """Discover real local LoRA adapter directories, preferring project outputs.

    The search intentionally ignores `.venv`, HuggingFace cache negative entries,
    and other dependency/cache folders. It also prioritizes the thesis project's
    known `outputs/lora/.../adapter` layout over smoke-test adapters.
    """
    adapters: List[Path] = []
    skipped_dir_names = {
        ".git", ".hg", ".svn", ".venv", "venv", "env", "__pycache__",
        "site-packages", "cache", ".cache", "huggingface", "hub", ".no_exist",
    }

    preferred_roots = [
        project_root / "outputs" / "lora",
        project_root / "outputs",
        project_root,
    ]

    seen = set()
    for root in preferred_roots:
        if not root.exists():
            continue
        for p in root.rglob("adapter_config.json"):
            parts_lower = {part.lower() for part in p.parts}
            if parts_lower.intersection(skipped_dir_names):
                continue
            adapter_dir = p.parent
            try:
                key = str(adapter_dir.resolve())
            except Exception:
                key = str(adapter_dir)
            if key in seen:
                continue
            if _is_valid_lora_adapter_dir(adapter_dir):
                seen.add(key)
                adapters.append(adapter_dir)

    def score(path: Path) -> Tuple[int, str]:
        s = str(path).lower().replace("\\", "/")
        # Prefer the main pilot run over smoke/check runs if all are present.
        priority = 100
        if "/outputs/lora/pmc_pilot_2000_r4_e2/adapter" in s:
            priority = 0
        elif "/outputs/lora/" in s and "smoke" not in s and "check" not in s:
            priority = 10
        elif "/outputs/lora/" in s:
            priority = 20
        return (priority, s)

    adapters.sort(key=score)
    return adapters


def env_summary() -> Dict[str, Any]:
    out: Dict[str, Any] = {"python": sys.version, "time": time.strftime("%Y-%m-%d %H:%M:%S")}
    try:
        import torch
        out["torch"] = torch.__version__
        out["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            out["cuda_device"] = torch.cuda.get_device_name(0)
            out["cuda_capability"] = torch.cuda.get_device_capability(0)
    except Exception as exc:
        out["torch_error"] = str(exc)
    try:
        import transformers
        out["transformers"] = transformers.__version__
    except Exception as exc:
        out["transformers_error"] = str(exc)
    try:
        import peft
        out["peft"] = peft.__version__
    except Exception as exc:
        out["peft_error"] = str(exc)
    return out
