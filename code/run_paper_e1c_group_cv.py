#!/usr/bin/env python3
"""PMCID-grouped cross-validation for paper ablations.

Uses only the cached 12,000-row training set and 1,000-row development set.
It never opens the 290-row holdout or 2,000-row canonical test.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import re
import sys
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
import run_paper_e1_feature_ablations as e1

PMCID_RE = re.compile(r"(PMC\d+)", re.IGNORECASE)
DEFAULT_EXPERIMENTS = (
    "E1_full",
    "E1_linear",
    "E1_no_zscore",
    "E1_pairs_only",
    "E1_biomed_only",
)


def pmcid(row: dict) -> str:
    match = PMCID_RE.search(str(row.get("image_ref", "")))
    if not match:
        raise ValueError(f"Could not extract PMCID from image_ref={row.get('image_ref')!r}")
    return match.group(1).upper()


def stable_tiebreak(text: str, seed: int) -> int:
    raw = hashlib.sha256(f"{seed}|{text}".encode("utf-8")).digest()
    return int.from_bytes(raw[:8], "big")


def make_group_folds(rows: list[dict], gold: torch.Tensor, n_folds: int, seed: int) -> np.ndarray:
    """Deterministic greedy assignment balancing row counts and four labels."""
    if n_folds < 2:
        raise ValueError("n_folds must be >= 2")
    groups: dict[str, list[int]] = defaultdict(list)
    for i, row in enumerate(rows):
        groups[pmcid(row)].append(i)
    if len(groups) < n_folds:
        raise ValueError(f"Only {len(groups)} PMCID groups for {n_folds} folds")

    gold_np = gold.cpu().numpy().astype(int)
    records = []
    for group_id, idxs in groups.items():
        counts = np.bincount(gold_np[idxs], minlength=4).astype(float)
        records.append((group_id, np.asarray(idxs, dtype=int), counts))
    records.sort(key=lambda x: (-len(x[1]), stable_tiebreak(x[0], seed)))

    total_n = len(rows)
    total_labels = np.bincount(gold_np, minlength=4).astype(float)
    target_n = total_n / n_folds
    target_labels = total_labels / n_folds
    fold_sizes = np.zeros(n_folds, dtype=float)
    fold_labels = np.zeros((n_folds, 4), dtype=float)
    group_to_fold: dict[str, int] = {}

    for pos, (group_id, idxs, counts) in enumerate(records):
        if pos < n_folds:
            candidates = [pos]
        else:
            candidates = list(range(n_folds))
        scored = []
        for fold in candidates:
            sizes = fold_sizes.copy()
            labels = fold_labels.copy()
            sizes[fold] += len(idxs)
            labels[fold] += counts
            size_cost = float(np.mean(((sizes - target_n) / max(target_n, 1.0)) ** 2))
            label_cost = float(np.mean(((labels - target_labels) / np.maximum(target_labels, 1.0)) ** 2))
            empty_penalty = float(np.sum(sizes == 0)) * 0.01
            score = size_cost + label_cost + empty_penalty
            scored.append((score, fold_sizes[fold], stable_tiebreak(f"{group_id}|{fold}", seed), fold))
        _, _, _, chosen = min(scored)
        group_to_fold[group_id] = chosen
        fold_sizes[chosen] += len(idxs)
        fold_labels[chosen] += counts

    assignments = np.asarray([group_to_fold[pmcid(row)] for row in rows], dtype=int)
    for fold in range(n_folds):
        if not np.any(assignments == fold):
            raise RuntimeError(f"Fold {fold} is empty")
    return assignments


def concat_pack(a: dict, b: dict) -> dict:
    return {
        "pair_features": torch.cat([a["pair_features"].float(), b["pair_features"].float()], dim=0),
        "base_scores": torch.cat([a["base_scores"].float(), b["base_scores"].float()], dim=0),
        "gold_indices": torch.cat([a["gold_indices"].long(), b["gold_indices"].long()], dim=0),
        "rows": list(a["rows"]) + list(b["rows"]),
        "source_split": ["train12000"] * len(a["rows"]) + ["dev1000"] * len(b["rows"]),
    }


def mean_std(values: list[float]) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    return float(arr.mean()), float(arr.std(ddof=1)) if len(arr) > 1 else 0.0


def main() -> None:
    ap = argparse.ArgumentParser(description="PMCID-grouped CV for frozen dual-encoder residual ablations")
    ap.add_argument("--root", type=Path, required=True,
                    help="Directory containing joint_train12000_dev1000_multiseed_v1")
    ap.add_argument("--e1-script-dir", type=Path, default=SCRIPT_DIR,
                    help="Directory containing run_paper_e1_feature_ablations.py")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--only", nargs="*", default=list(DEFAULT_EXPERIMENTS))
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--fold-seed", type=int, default=20260801)
    ap.add_argument("--seeds", nargs="*", type=int, default=list(e1.SEEDS))
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=15)
    args = ap.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif args.device == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("CUDA requested but unavailable")
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))

    tr = args.root / "joint_train12000_dev1000_multiseed_v1"
    tb = e1.load(tr / "train_biomed_features.pt")
    tc = e1.load(tr / "train_clip_features.pt")
    vb = e1.load(tr / "val_biomed_features.pt")
    vc = e1.load(tr / "val_clip_features.pt")
    if not (torch.equal(tb["gold_indices"], tc["gold_indices"]) and
            torch.equal(vb["gold_indices"], vc["gold_indices"])):
        raise ValueError("BiomedCLIP and CLIP caches are not aligned")

    biomed = concat_pack(tb, vb)
    clip = concat_pack(tc, vc)
    if not torch.equal(biomed["gold_indices"], clip["gold_indices"]):
        raise ValueError("Combined caches are not aligned")
    rows = biomed["rows"]
    gold = biomed["gold_indices"]
    fold_ids = make_group_folds(rows, gold, args.folds, args.fold_seed)

    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "group_fold_manifest.csv").open("w", encoding="utf-8", newline="") as f:
        fields = ["row_index", "source_split", "pmcid", "gold_index", "fold"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for i, row in enumerate(rows):
            w.writerow({"row_index": i, "source_split": biomed["source_split"][i],
                        "pmcid": pmcid(row), "gold_index": int(gold[i]), "fold": int(fold_ids[i])})

    spec_map = {spec.name: spec for spec in e1.SPECS}
    unknown = sorted(set(args.only) - set(spec_map))
    if unknown:
        raise ValueError(f"Unknown experiment IDs: {unknown}")
    specs = [spec_map[name] for name in args.only]

    all_fold_rows: list[dict] = []
    aggregate_rows: list[dict] = []
    started_all = time.time()

    for spec in specs:
        print(f"START {spec.name}", flush=True)
        oof_pred = torch.full((len(gold),), -1, dtype=torch.long)
        oof_base = torch.full((len(gold),), -1, dtype=torch.long)
        fold_accs: list[float] = []
        fold_base_accs: list[float] = []
        exp_started = time.time()

        for fold in range(args.folds):
            val_mask_np = fold_ids == fold
            train_mask_np = ~val_mask_np
            train_idx = torch.from_numpy(np.flatnonzero(train_mask_np)).long()
            val_idx = torch.from_numpy(np.flatnonzero(val_mask_np)).long()

            tbp = biomed["pair_features"][train_idx]
            tcp = clip["pair_features"][train_idx]
            tbs = biomed["base_scores"][train_idx]
            tcs = clip["base_scores"][train_idx]
            norm = e1.Normalizer(tbp, tcp, tbs, tcs)

            tx = e1.build_features(spec, tbp, tcp, tbs, tcs, norm)
            vx = e1.build_features(spec,
                                   biomed["pair_features"][val_idx], clip["pair_features"][val_idx],
                                   biomed["base_scores"][val_idx], clip["base_scores"][val_idx], norm)
            if spec.base == "biomed":
                tbase = tbs
                vbase = biomed["base_scores"][val_idx]
            else:
                tbase = tcs
                vbase = clip["base_scores"][val_idx]
            tgold = gold[train_idx]
            vgold = gold[val_idx]
            base_pred = vbase.argmax(1)
            base_acc = float(base_pred.eq(vgold).float().mean())

            seed_logits = []
            seed_accs = []
            seed_epochs = []
            for seed in args.seeds:
                model, sr = e1.train_seed(
                    spec, seed, tx, tbase, tgold, vx, vbase, vgold, device,
                    epochs=args.epochs, patience=args.patience, batch=args.batch_size,
                )
                _, _, logits = e1.eval_model(model, vx, vbase, vgold, args.batch_size, device)
                seed_logits.append(logits)
                seed_accs.append(float(logits.argmax(1).eq(vgold).float().mean()))
                seed_epochs.append(int(sr["best_epoch"]))

            ensemble_logits = torch.stack(seed_logits).mean(0)
            pred = ensemble_logits.argmax(1)
            acc = float(pred.eq(vgold).float().mean())
            paired = e1.paired(vgold, base_pred, pred)
            oof_pred[val_idx] = pred
            oof_base[val_idx] = base_pred
            fold_accs.append(acc)
            fold_base_accs.append(base_acc)
            all_fold_rows.append({
                "experiment_id": spec.name,
                "fold": fold,
                "n_train": int(len(train_idx)),
                "n_validation": int(len(val_idx)),
                "unique_pmcid_validation": len({pmcid(rows[i]) for i in val_idx.tolist()}),
                "base_accuracy": base_acc,
                "ensemble_accuracy": acc,
                "delta": acc - base_acc,
                "fixed": paired["fixed"],
                "broken": paired["broken"],
                "net": paired["net"],
                "mcnemar_exact_p": paired["mcnemar_exact_p"],
                "seed_accuracies": ";".join(f"{x:.6f}" for x in seed_accs),
                "best_epochs": ";".join(str(x) for x in seed_epochs),
            })
            print(f"  fold={fold} n={len(val_idx)} acc={acc:.4f} base={base_acc:.4f}", flush=True)

        if int((oof_pred < 0).sum()) or int((oof_base < 0).sum()):
            raise RuntimeError(f"Missing OOF predictions for {spec.name}")
        overall_acc = float(oof_pred.eq(gold).float().mean())
        overall_base = float(oof_base.eq(gold).float().mean())
        paired = e1.paired(gold, oof_base, oof_pred)
        mean_acc, std_acc = mean_std(fold_accs)
        mean_base, std_base = mean_std(fold_base_accs)
        params = sum(p.numel() for p in e1.Residual(
            e1.build_features(
                spec,
                biomed["pair_features"][:1], clip["pair_features"][:1],
                biomed["base_scores"][:1], clip["base_scores"][:1],
                e1.Normalizer(
                    biomed["pair_features"][:100], clip["pair_features"][:100],
                    biomed["base_scores"][:100], clip["base_scores"][:100],
                ),
            ).shape[-1], spec
        ).parameters())
        aggregate_rows.append({
            "experiment_id": spec.name,
            "n": len(gold),
            "n_folds": args.folds,
            "n_seeds": len(args.seeds),
            "trainable_parameters": params,
            "oof_base_accuracy": overall_base,
            "oof_accuracy": overall_acc,
            "oof_delta": overall_acc - overall_base,
            "fold_accuracy_mean": mean_acc,
            "fold_accuracy_std": std_acc,
            "fold_base_accuracy_mean": mean_base,
            "fold_base_accuracy_std": std_base,
            "fixed": paired["fixed"],
            "broken": paired["broken"],
            "net": paired["net"],
            "mcnemar_exact_p": paired["mcnemar_exact_p"],
            "runtime_seconds": time.time() - exp_started,
        })

        with (args.out / f"{spec.name}_oof_predictions.csv").open("w", encoding="utf-8", newline="") as f:
            fields = ["row_index", "source_split", "pmcid", "fold", "gold_index",
                      "base_pred", "new_pred", "base_correct", "new_correct"]
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for i in range(len(gold)):
                w.writerow({
                    "row_index": i,
                    "source_split": biomed["source_split"][i],
                    "pmcid": pmcid(rows[i]),
                    "fold": int(fold_ids[i]),
                    "gold_index": int(gold[i]),
                    "base_pred": int(oof_base[i]),
                    "new_pred": int(oof_pred[i]),
                    "base_correct": int(oof_base[i] == gold[i]),
                    "new_correct": int(oof_pred[i] == gold[i]),
                })
        print(f"DONE {spec.name} OOF={overall_acc:.4f} base={overall_base:.4f}", flush=True)

    with (args.out / "group_cv_fold_results.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_fold_rows[0]))
        w.writeheader(); w.writerows(all_fold_rows)
    with (args.out / "group_cv_summary.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(aggregate_rows[0]))
        w.writeheader(); w.writerows(aggregate_rows)

    payload = {
        "protocol": {
            "description": "Exploratory PMCID-grouped cross-validation on train12000 + dev1000 only",
            "holdout_or_test_opened": False,
            "n_rows": len(gold),
            "unique_pmcid": len({pmcid(r) for r in rows}),
            "folds": args.folds,
            "fold_seed": args.fold_seed,
            "seeds": args.seeds,
            "experiments": args.only,
            "device": str(device),
            "torch_version": torch.__version__,
            "cuda_device": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
            "runtime_seconds": time.time() - started_all,
        },
        "results": aggregate_rows,
        "interpretation_guardrail": (
            "Post-hoc exploratory model-class analysis. Do not use these results to replace the "
            "pre-specified full model or reopen the canonical test."
        ),
    }
    with (args.out / "group_cv_summary.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"WROTE {args.out / 'group_cv_summary.csv'}", flush=True)


if __name__ == "__main__":
    main()
