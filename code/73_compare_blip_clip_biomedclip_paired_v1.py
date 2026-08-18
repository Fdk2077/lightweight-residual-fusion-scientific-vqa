#!/usr/bin/env python3
"""Paired three-model comparison on the canonical PMC-VQA clean 2,000 set.

Models:
- canonical BLIP baseline
- general OpenAI CLIP ViT-B/16
- BiomedCLIP

The script aligns rows by (image basename, normalized question), computes
paired accuracy differences, exact two-sided McNemar tests, paired bootstrap
confidence intervals, Holm-adjusted p-values, agreement patterns, and a small
audit of CLIP text truncation.

Run from D:/thesis-vqa.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--blip-biomedclip-paired",
        type=Path,
        default=Path(
            "outputs/model_comparison/"
            "blip_vs_biomedclip_clean2000_v1/"
            "blip_vs_biomedclip_paired.csv"
        ),
    )
    p.add_argument(
        "--clip-predictions",
        type=Path,
        default=Path(
            "outputs/clip/"
            "clip_vitb16_clean2000_question_answer_frozen_v1/"
            "clip_predictions.csv"
        ),
    )
    p.add_argument(
        "--clip-template",
        default="question_answer",
    )
    p.add_argument("--bootstrap-replicates", type=int, default=50000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path(
            "outputs/model_comparison/"
            "blip_clip_biomedclip_clean2000_v1"
        ),
    )
    return p.parse_args()


def read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def image_basename(value: str) -> str:
    value = (value or "").replace("\\", "/")
    return value.rsplit("/", 1)[-1].strip()


def normalize_question(value: str) -> str:
    return re.sub(r"\s+", " ", (value or "").strip())


def normalize_label(value: Any) -> str:
    text = str(value or "").strip().upper()
    match = re.match(r"^([ABCD])", text)
    if match:
        return match.group(1)
    if text in {"0", "1", "2", "3"}:
        return "ABCD"[int(text)]
    return text


def bool_int(value: Any) -> int:
    text = str(value).strip().lower()
    if text in {"1", "true", "yes"}:
        return 1
    if text in {"0", "false", "no", ""}:
        return 0
    return int(float(text))


def row_key(row: Mapping[str, str]) -> Tuple[str, str]:
    image = (
        row.get("image_ref")
        or row.get("image")
        or row.get("image_path")
        or ""
    )
    return image_basename(image), normalize_question(row.get("question", ""))


def build_unique_map(
    rows: Sequence[Dict[str, str]],
    name: str,
) -> Dict[Tuple[str, str], Dict[str, str]]:
    mapping: Dict[Tuple[str, str], Dict[str, str]] = {}
    duplicates: List[Tuple[str, str]] = []
    for row in rows:
        key = row_key(row)
        if key in mapping:
            duplicates.append(key)
        mapping[key] = row
    if duplicates:
        raise ValueError(f"{name} contains {len(duplicates)} duplicate keys.")
    return mapping


def exact_mcnemar_p(fixed_by_b: int, broken_by_b: int) -> float:
    n = fixed_by_b + broken_by_b
    if n == 0:
        return 1.0
    k = min(fixed_by_b, broken_by_b)
    lower_tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2.0 * lower_tail)


def paired_bootstrap_ci(
    differences: Sequence[float],
    replicates: int,
    seed: int,
) -> Tuple[float, float]:
    if not differences:
        return 0.0, 0.0
    if replicates <= 0:
        point = sum(differences) / len(differences)
        return point, point

    rng = random.Random(seed)
    n = len(differences)
    means: List[float] = []
    for _ in range(replicates):
        means.append(
            sum(differences[rng.randrange(n)] for _ in range(n)) / n
        )
    means.sort()
    low_index = max(0, int(0.025 * replicates))
    high_index = min(replicates - 1, int(0.975 * replicates))
    return means[low_index], means[high_index]


def holm_adjust(p_values: Mapping[str, float]) -> Dict[str, float]:
    ordered = sorted(p_values.items(), key=lambda item: item[1])
    m = len(ordered)
    adjusted: Dict[str, float] = {}
    running_max = 0.0
    for rank, (name, p_value) in enumerate(ordered):
        multiplier = m - rank
        candidate = min(1.0, p_value * multiplier)
        running_max = max(running_max, candidate)
        adjusted[name] = running_max
    return adjusted


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    args.blip_biomedclip_paired = args.blip_biomedclip_paired.resolve()
    args.clip_predictions = args.clip_predictions.resolve()
    args.out_dir = args.out_dir.resolve()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    paired_rows = read_csv(args.blip_biomedclip_paired)
    clip_rows = read_csv(args.clip_predictions)

    paired_map = build_unique_map(paired_rows, "BLIP/BiomedCLIP file")
    clip_map = build_unique_map(clip_rows, "CLIP file")

    paired_keys = set(paired_map)
    clip_keys = set(clip_map)
    common_keys = paired_keys & clip_keys

    if len(common_keys) != 2000:
        raise ValueError(
            "Expected 2,000 aligned rows, but found "
            f"{len(common_keys)}. "
            f"paired_only={len(paired_keys - clip_keys)}, "
            f"clip_only={len(clip_keys - paired_keys)}"
        )

    clip_pred_column = f"{args.clip_template}_pred_label"
    clip_correct_column = f"{args.clip_template}_correct"
    clip_trunc_column = f"{args.clip_template}_sample_had_truncation"

    merged: List[Dict[str, Any]] = []
    gold_mismatches = 0
    saved_correctness_mismatches = Counter()

    for key in sorted(common_keys):
        paired = paired_map[key]
        clip = clip_map[key]

        gold_a = normalize_label(paired.get("gold_label"))
        gold_b = normalize_label(clip.get("gold_label"))
        if gold_a != gold_b:
            gold_mismatches += 1

        gold = gold_a
        blip_pred = normalize_label(paired.get("blip_pred_label"))
        clip_pred = normalize_label(clip.get(clip_pred_column))
        biomedclip_pred = normalize_label(paired.get("biomedclip_pred_label"))

        blip_correct = int(blip_pred == gold)
        clip_correct = int(clip_pred == gold)
        biomedclip_correct = int(biomedclip_pred == gold)

        if "blip_correct" in paired:
            if blip_correct != bool_int(paired["blip_correct"]):
                saved_correctness_mismatches["blip"] += 1
        if clip_correct_column in clip:
            if clip_correct != bool_int(clip[clip_correct_column]):
                saved_correctness_mismatches["clip"] += 1
        if "biomedclip_correct" in paired:
            if biomedclip_correct != bool_int(paired["biomedclip_correct"]):
                saved_correctness_mismatches["biomedclip"] += 1

        truncated = (
            bool_int(clip.get(clip_trunc_column, "0"))
            if clip_trunc_column in clip
            else 0
        )

        merged.append(
            {
                "image_ref": key[0],
                "question": key[1],
                "gold_label": gold,
                "blip_pred_label": blip_pred,
                "clip_pred_label": clip_pred,
                "biomedclip_pred_label": biomedclip_pred,
                "blip_correct": blip_correct,
                "clip_correct": clip_correct,
                "biomedclip_correct": biomedclip_correct,
                "clip_sample_had_truncation": truncated,
                "choice_A": paired.get("choice_A", clip.get("choice_A", "")),
                "choice_B": paired.get("choice_B", clip.get("choice_B", "")),
                "choice_C": paired.get("choice_C", clip.get("choice_C", "")),
                "choice_D": paired.get("choice_D", clip.get("choice_D", "")),
            }
        )

    if gold_mismatches:
        raise ValueError(f"Gold-label mismatches: {gold_mismatches}")

    model_columns = {
        "BLIP": "blip_correct",
        "CLIP": "clip_correct",
        "BiomedCLIP": "biomedclip_correct",
    }

    single_results: Dict[str, Any] = {}
    for model, column in model_columns.items():
        correct = [int(row[column]) for row in merged]
        single_results[model] = {
            "samples": len(correct),
            "correct": sum(correct),
            "accuracy": sum(correct) / len(correct),
        }

    pairs = [
        ("BLIP", "CLIP"),
        ("BLIP", "BiomedCLIP"),
        ("CLIP", "BiomedCLIP"),
    ]

    pairwise: Dict[str, Any] = {}
    raw_p_values: Dict[str, float] = {}

    for model_a, model_b in pairs:
        column_a = model_columns[model_a]
        column_b = model_columns[model_b]
        values_a = [int(row[column_a]) for row in merged]
        values_b = [int(row[column_b]) for row in merged]
        differences = [
            float(value_b - value_a)
            for value_a, value_b in zip(values_a, values_b)
        ]

        both_correct = sum(a == 1 and b == 1 for a, b in zip(values_a, values_b))
        both_wrong = sum(a == 0 and b == 0 for a, b in zip(values_a, values_b))
        fixed_by_b = sum(a == 0 and b == 1 for a, b in zip(values_a, values_b))
        broken_by_b = sum(a == 1 and b == 0 for a, b in zip(values_a, values_b))

        p_value = exact_mcnemar_p(fixed_by_b, broken_by_b)
        raw_p_values[f"{model_a}_vs_{model_b}"] = p_value
        ci_low, ci_high = paired_bootstrap_ci(
            differences,
            replicates=args.bootstrap_replicates,
            seed=args.seed,
        )

        pairwise[f"{model_a}_vs_{model_b}"] = {
            "model_a": model_a,
            "model_b": model_b,
            "accuracy_a": sum(values_a) / len(values_a),
            "accuracy_b": sum(values_b) / len(values_b),
            "delta_b_minus_a": sum(differences) / len(differences),
            "paired_bootstrap_ci95_low": ci_low,
            "paired_bootstrap_ci95_high": ci_high,
            "both_correct": both_correct,
            "both_wrong": both_wrong,
            "fixed_by_b": fixed_by_b,
            "broken_by_b": broken_by_b,
            "exact_mcnemar_p": p_value,
        }

    holm = holm_adjust(raw_p_values)
    for name, adjusted_p in holm.items():
        pairwise[name]["holm_adjusted_p"] = adjusted_p

    correctness_patterns = Counter()
    prediction_agreement = Counter()

    for row in merged:
        correctness_pattern = (
            f"B{row['blip_correct']}_"
            f"C{row['clip_correct']}_"
            f"M{row['biomedclip_correct']}"
        )
        correctness_patterns[correctness_pattern] += 1

        labels = {
            row["blip_pred_label"],
            row["clip_pred_label"],
            row["biomedclip_pred_label"],
        }
        if len(labels) == 1:
            prediction_agreement["all_same_label"] += 1
        elif row["blip_pred_label"] == row["clip_pred_label"]:
            prediction_agreement["blip_clip_same_only"] += 1
        elif row["blip_pred_label"] == row["biomedclip_pred_label"]:
            prediction_agreement["blip_biomedclip_same_only"] += 1
        elif row["clip_pred_label"] == row["biomedclip_pred_label"]:
            prediction_agreement["clip_biomedclip_same_only"] += 1
        else:
            prediction_agreement["all_three_different"] += 1

    truncated_rows = [
        row for row in merged if row["clip_sample_had_truncation"] == 1
    ]
    nontruncated_rows = [
        row for row in merged if row["clip_sample_had_truncation"] == 0
    ]

    def subgroup_accuracy(rows: Sequence[Mapping[str, Any]], column: str) -> float:
        if not rows:
            return 0.0
        return sum(int(row[column]) for row in rows) / len(rows)

    truncation_audit = {
        "truncated_samples": len(truncated_rows),
        "nontruncated_samples": len(nontruncated_rows),
        "truncated_sample_accuracies": {
            model: subgroup_accuracy(truncated_rows, column)
            for model, column in model_columns.items()
        },
        "nontruncated_sample_accuracies": {
            model: subgroup_accuracy(nontruncated_rows, column)
            for model, column in model_columns.items()
        },
        "note": (
            "The truncated subgroup is descriptive only because it contains "
            "very few samples."
        ),
    }

    label_distributions = {
        "BLIP": dict(Counter(row["blip_pred_label"] for row in merged)),
        "CLIP": dict(Counter(row["clip_pred_label"] for row in merged)),
        "BiomedCLIP": dict(
            Counter(row["biomedclip_pred_label"] for row in merged)
        ),
        "gold": dict(Counter(row["gold_label"] for row in merged)),
    }

    summary = {
        "run_name": args.out_dir.name,
        "samples": len(merged),
        "alignment": {
            "paired_file_rows": len(paired_rows),
            "clip_file_rows": len(clip_rows),
            "shared_rows": len(common_keys),
            "paired_only": len(paired_keys - clip_keys),
            "clip_only": len(clip_keys - paired_keys),
            "gold_label_mismatches": gold_mismatches,
            "saved_correctness_mismatches": dict(
                saved_correctness_mismatches
            ),
        },
        "single_model_results": single_results,
        "pairwise_comparisons": pairwise,
        "holm_family": list(raw_p_values),
        "correctness_patterns": dict(correctness_patterns),
        "prediction_agreement": dict(prediction_agreement),
        "label_distributions": label_distributions,
        "clip_truncation_audit": truncation_audit,
        "bootstrap_replicates": args.bootstrap_replicates,
        "seed": args.seed,
        "interpretation_note": (
            "CLIP and BiomedCLIP are compared under the same frozen "
            "question_answer prompt on the same canonical 2,000 samples. "
            "BiomedCLIP and general CLIP share a ViT-B/16-scale visual "
            "backbone family, but their text encoders and pretraining data "
            "are not identical."
        ),
    }

    write_csv(args.out_dir / "three_model_paired_predictions.csv", merged)
    (args.out_dir / "three_model_comparison_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    report_lines = [
        "BLIP vs CLIP vs BiomedCLIP — canonical PMC-VQA clean 2,000",
        "=" * 78,
        "",
        "SINGLE MODEL RESULTS",
    ]
    for model, result in single_results.items():
        report_lines.append(
            f"{model}: {result['accuracy']:.4f} "
            f"({result['correct']}/{result['samples']})"
        )

    report_lines.extend(["", "PAIRED COMPARISONS"])
    for name, result in pairwise.items():
        report_lines.append(
            f"{name}: delta={result['delta_b_minus_a']:+.4f}, "
            f"CI95=[{result['paired_bootstrap_ci95_low']:+.4f}, "
            f"{result['paired_bootstrap_ci95_high']:+.4f}], "
            f"fixed/broken={result['fixed_by_b']}/"
            f"{result['broken_by_b']}, "
            f"McNemar p={result['exact_mcnemar_p']:.12g}, "
            f"Holm p={result['holm_adjusted_p']:.12g}"
        )

    report_lines.extend([
        "",
        "CLIP TRUNCATION AUDIT",
        json.dumps(truncation_audit, ensure_ascii=False, indent=2),
        "",
        "CORRECTNESS PATTERNS",
        json.dumps(dict(correctness_patterns), ensure_ascii=False, indent=2),
    ])

    (args.out_dir / "REPORT.txt").write_text(
        "\n".join(report_lines),
        encoding="utf-8",
    )

    print("\n".join(report_lines))
    print("")
    print(
        "Summary:",
        args.out_dir / "three_model_comparison_summary.json",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
