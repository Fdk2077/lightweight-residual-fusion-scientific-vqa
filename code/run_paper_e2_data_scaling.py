#!/usr/bin/env python3
"""Nested PMCID-disjoint data-scaling study for the paper.

Uses cached frozen CLIP/BiomedCLIP features from train12000 and dev1000 only.
All training rows whose PMCID appears in dev1000 are excluded. The remaining
training groups are deterministically ordered to form nested, approximately
label-balanced subsets. The script never opens holdout290 or canonical test2000.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
import run_paper_e1_feature_ablations as e1

PMCID_RE = re.compile(r"(PMC\d+)", re.IGNORECASE)
DEFAULT_MODELS = ("E1_full", "E1_linear")
DEFAULT_TARGETS = (2000, 4000, 8000)


def pmcid(row: dict) -> str:
    match = PMCID_RE.search(str(row.get("image_ref", "")))
    if not match:
        raise ValueError(f"Could not extract PMCID from image_ref={row.get('image_ref')!r}")
    return match.group(1).upper()


def stable_hash(text: str, seed: int) -> int:
    raw = hashlib.sha256(f"{seed}|{text}".encode("utf-8")).digest()
    return int.from_bytes(raw[:8], "big")


def build_nested_group_order(rows: list[dict], gold: torch.Tensor, excluded_pmcids: set[str], seed: int) -> tuple[list[int], list[dict]]:
    """Return a deterministic nested row order with PMCID groups kept intact.

    Groups are assigned a primary label by their largest label count. A deficit
    scheduler chooses the next primary-label queue so that prefixes track the
    eligible-set label proportions. Within each queue, groups are stably shuffled.
    """
    groups: dict[str, list[int]] = defaultdict(list)
    for i, row in enumerate(rows):
        gid = pmcid(row)
        if gid not in excluded_pmcids:
            groups[gid].append(i)
    if not groups:
        raise ValueError("No eligible training PMCID groups remain")

    gold_np = gold.cpu().numpy().astype(int)
    records: list[dict] = []
    total_labels = np.zeros(4, dtype=float)
    for gid, idxs in groups.items():
        counts = np.bincount(gold_np[idxs], minlength=4).astype(int)
        total_labels += counts
        max_count = int(counts.max())
        tied = np.flatnonzero(counts == max_count).tolist()
        primary = min(tied, key=lambda lab: stable_hash(f"{gid}|label{lab}", seed))
        records.append({"pmcid": gid, "indices": idxs, "counts": counts, "primary": int(primary)})

    queues: dict[int, list[dict]] = {lab: [] for lab in range(4)}
    for rec in records:
        queues[rec["primary"]].append(rec)
    for lab in range(4):
        queues[lab].sort(key=lambda r: (stable_hash(r["pmcid"], seed), r["pmcid"]))

    target_prop = total_labels / total_labels.sum()
    selected_labels = np.zeros(4, dtype=float)
    selected_n = 0
    ordered_records: list[dict] = []
    positions = {lab: 0 for lab in range(4)}

    while len(ordered_records) < len(records):
        candidates = []
        for lab in range(4):
            pos = positions[lab]
            if pos >= len(queues[lab]):
                continue
            rec = queues[lab][pos]
            next_n = selected_n + len(rec["indices"])
            after = selected_labels + rec["counts"]
            # Mean squared relative deviation from the desired cumulative counts.
            desired = target_prop * next_n
            balance_cost = float(np.mean(((after - desired) / np.maximum(desired, 1.0)) ** 2))
            candidates.append((balance_cost, stable_hash(rec["pmcid"], seed), lab, rec))
        if not candidates:
            break
        _, _, lab, rec = min(candidates)
        positions[lab] += 1
        ordered_records.append(rec)
        selected_labels += rec["counts"]
        selected_n += len(rec["indices"])

    row_order: list[int] = []
    group_manifest: list[dict] = []
    cumulative = 0
    for order, rec in enumerate(ordered_records):
        row_order.extend(rec["indices"])
        cumulative += len(rec["indices"])
        group_manifest.append({
            "group_order": order,
            "pmcid": rec["pmcid"],
            "group_rows": len(rec["indices"]),
            "cumulative_rows": cumulative,
            "label_0": int(rec["counts"][0]),
            "label_1": int(rec["counts"][1]),
            "label_2": int(rec["counts"][2]),
            "label_3": int(rec["counts"][3]),
            "primary_label": int(rec["primary"]),
        })
    return row_order, group_manifest


def subset_for_target(row_order: list[int], rows: list[dict], target: int) -> list[int]:
    """Take the shortest PMCID-complete prefix reaching target rows."""
    if target <= 0:
        raise ValueError("Target size must be positive")
    selected: list[int] = []
    seen: set[str] = set()
    for idx in row_order:
        gid = pmcid(rows[idx])
        if gid in seen:
            continue
        seen.add(gid)
        # Add the whole group from the global order.
        group_idxs = [j for j in row_order if pmcid(rows[j]) == gid]
        selected.extend(group_idxs)
        if len(selected) >= target:
            break
    return selected


def efficient_prefix_subsets(row_order: list[int], rows: list[dict], targets: list[int]) -> dict[int, list[int]]:
    """Build PMCID-complete nested prefixes in one pass."""
    sorted_targets = sorted(set(targets))
    out: dict[int, list[int]] = {}
    current: list[int] = []
    pos = 0
    while pos < len(row_order) and sorted_targets:
        gid = pmcid(rows[row_order[pos]])
        group: list[int] = []
        while pos < len(row_order) and pmcid(rows[row_order[pos]]) == gid:
            group.append(row_order[pos])
            pos += 1
        current.extend(group)
        while sorted_targets and len(current) >= sorted_targets[0]:
            out[sorted_targets.pop(0)] = list(current)
    for target in sorted_targets:
        out[target] = list(current)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Nested PMCID-disjoint data-scaling study")
    ap.add_argument("--root", type=Path, required=True,
                    help="Directory containing joint_train12000_dev1000_multiseed_v1")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--only", nargs="*", default=list(DEFAULT_MODELS),
                    help="Model IDs from the E1 script; default: E1_full E1_linear")
    ap.add_argument("--targets", nargs="*", type=int, default=list(DEFAULT_TARGETS),
                    help="Requested nested training sizes; the full eligible set is always added")
    ap.add_argument("--subset-seed", type=int, default=20260801)
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

    train_rows = list(tb["rows"])
    dev_rows = list(vb["rows"])
    train_gold = tb["gold_indices"].long()
    dev_gold = vb["gold_indices"].long()
    dev_pmcids = {pmcid(row) for row in dev_rows}
    overlap_rows = sum(pmcid(row) in dev_pmcids for row in train_rows)

    row_order, group_manifest = build_nested_group_order(
        train_rows, train_gold, excluded_pmcids=dev_pmcids, seed=args.subset_seed
    )
    eligible_n = len(row_order)
    requested = sorted(set(x for x in args.targets if 0 < x < eligible_n))
    requested.append(eligible_n)
    subsets = efficient_prefix_subsets(row_order, train_rows, requested)

    spec_map = {spec.name: spec for spec in e1.SPECS}
    unknown = sorted(set(args.only) - set(spec_map))
    if unknown:
        raise ValueError(f"Unknown model IDs: {unknown}")
    specs = [spec_map[name] for name in args.only]

    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "nested_group_order.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(group_manifest[0]))
        w.writeheader(); w.writerows(group_manifest)

    subset_rows: list[dict] = []
    result_rows: list[dict] = []
    summaries: dict[str, dict] = {}
    started_all = time.time()

    dev_base_biomed = vb["base_scores"].float()
    dev_base_clip = vc["base_scores"].float()

    for target in requested:
        idx_list = subsets[target]
        idx = torch.tensor(idx_list, dtype=torch.long)
        actual_n = len(idx_list)
        subset_pmcids = {pmcid(train_rows[i]) for i in idx_list}
        labels = np.bincount(train_gold[idx].cpu().numpy(), minlength=4)
        subset_rows.append({
            "requested_rows": target,
            "actual_rows": actual_n,
            "unique_pmcid": len(subset_pmcids),
            "dev_pmcid_overlap": len(subset_pmcids & dev_pmcids),
            "label_0": int(labels[0]), "label_1": int(labels[1]),
            "label_2": int(labels[2]), "label_3": int(labels[3]),
        })

        tbp = tb["pair_features"][idx].float()
        tcp = tc["pair_features"][idx].float()
        tbs = tb["base_scores"][idx].float()
        tcs = tc["base_scores"][idx].float()
        tgold = train_gold[idx]
        norm = e1.Normalizer(tbp, tcp, tbs, tcs)

        for spec in specs:
            tag = f"{spec.name}_n{actual_n}"
            print(f"START {tag}", flush=True)
            started = time.time()
            tx = e1.build_features(spec, tbp, tcp, tbs, tcs, norm)
            vx = e1.build_features(
                spec,
                vb["pair_features"].float(), vc["pair_features"].float(),
                vb["base_scores"].float(), vc["base_scores"].float(), norm,
            )
            if spec.base == "biomed":
                tbase, vbase = tbs, dev_base_biomed
            else:
                tbase, vbase = tcs, dev_base_clip
            base_pred = vbase.argmax(1)
            base_acc = float(base_pred.eq(dev_gold).float().mean())

            models = []
            seed_logits = []
            seed_results = []
            for seed in args.seeds:
                model, sr = e1.train_seed(
                    spec, seed, tx, tbase, tgold, vx, vbase, dev_gold, device,
                    epochs=args.epochs, patience=args.patience, batch=args.batch_size,
                )
                _, _, logits = e1.eval_model(model, vx, vbase, dev_gold, args.batch_size, device)
                models.append(model); seed_logits.append(logits); seed_results.append(sr)
                result_rows.append({
                    "model": spec.name, "requested_rows": target, "actual_rows": actual_n,
                    "seed": seed, "dev_accuracy": float(logits.argmax(1).eq(dev_gold).float().mean()),
                    "base_accuracy": base_acc, "delta": float(logits.argmax(1).eq(dev_gold).float().mean()) - base_acc,
                    "best_epoch": int(sr["best_epoch"]), "trainable_parameters": sum(p.numel() for p in model.parameters()),
                })

            ensemble_logits = torch.stack(seed_logits).mean(0)
            pred = ensemble_logits.argmax(1)
            acc = float(pred.eq(dev_gold).float().mean())
            paired = e1.paired(dev_gold, base_pred, pred)
            result_rows.append({
                "model": spec.name, "requested_rows": target, "actual_rows": actual_n,
                "seed": "ensemble", "dev_accuracy": acc, "base_accuracy": base_acc,
                "delta": acc - base_acc, "best_epoch": "",
                "trainable_parameters": sum(p.numel() for p in models[0].parameters()),
            })
            summaries[tag] = {
                "model": spec.name,
                "spec": asdict(spec),
                "requested_rows": target,
                "actual_rows": actual_n,
                "unique_pmcid": len(subset_pmcids),
                "dev_pmcid_overlap": len(subset_pmcids & dev_pmcids),
                "input_dim": int(tx.shape[-1]),
                "trainable_parameters": sum(p.numel() for p in models[0].parameters()),
                "seed_results": seed_results,
                "dev": {"base_accuracy": base_acc, "ensemble_accuracy": acc,
                        "delta": acc - base_acc, "paired": paired},
                "runtime_seconds": time.time() - started,
            }
            expdir = args.out / tag
            expdir.mkdir(exist_ok=True)
            torch.save({"spec": asdict(spec), "state_dicts": [m.state_dict() for m in models]},
                       expdir / "ensemble_states.pt")
            with (expdir / "dev_predictions.csv").open("w", encoding="utf-8", newline="") as f:
                fields = ["row_index", "pmcid", "gold_index", "base_pred", "new_pred", "base_correct", "new_correct"]
                w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
                for i, (row, g, b, n) in enumerate(zip(dev_rows, dev_gold.tolist(), base_pred.tolist(), pred.tolist())):
                    w.writerow({"row_index": i, "pmcid": pmcid(row), "gold_index": g,
                                "base_pred": b, "new_pred": n,
                                "base_correct": int(b == g), "new_correct": int(n == g)})
            print(f"DONE {tag} dev={acc:.4f} base={base_acc:.4f} sec={time.time()-started:.1f}", flush=True)

    with (args.out / "subset_summary.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(subset_rows[0])); w.writeheader(); w.writerows(subset_rows)
    with (args.out / "data_scaling_results.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(result_rows[0])); w.writeheader(); w.writerows(result_rows)

    protocol = {
        "description": "Nested PMCID-disjoint data-scaling study on train12000 -> dev1000",
        "holdout_or_test_opened": False,
        "original_train_rows": len(train_rows),
        "original_dev_rows": len(dev_rows),
        "train_rows_excluded_for_dev_pmcid_overlap": overlap_rows,
        "eligible_train_rows": eligible_n,
        "eligible_unique_pmcid": len({pmcid(train_rows[i]) for i in row_order}),
        "dev_unique_pmcid": len(dev_pmcids),
        "requested_targets": requested,
        "subset_seed": args.subset_seed,
        "seeds": args.seeds,
        "models": args.only,
        "device": str(device),
        "torch_version": torch.__version__,
        "cuda_device": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "runtime_seconds": time.time() - started_all,
        "interpretation_guardrail": "Descriptive data-efficiency analysis. Do not use dev results to reopen holdout or canonical test.",
    }
    with (args.out / "data_scaling_summary.json").open("w", encoding="utf-8") as f:
        json.dump({"protocol": protocol, "subsets": subset_rows, "results": summaries}, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
