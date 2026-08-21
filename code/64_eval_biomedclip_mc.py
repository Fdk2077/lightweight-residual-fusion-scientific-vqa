#!/usr/bin/env python3
"""Zero-shot multiple-choice PMC-VQA evaluation with Microsoft BiomedCLIP.

The script compares prompt templates on validation data. After selecting a
prompt on validation, rerun with --frozen-template on the clean test set.
Options are cleaned by common_vqa.py, so positional prefixes such as A:/B:
are not included in the scored text.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import torch
from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from common_vqa import (  # type: ignore
    UserFacingError,
    bootstrap_ci_binary,
    clean_options,
    discover_image_roots,
    env_summary,
    get_answer_label,
    get_image_ref,
    get_options,
    get_question,
    index_images,
    log,
    read_csv_rows,
    resolve_image_path,
    write_csv_rows,
    write_json,
)

MODEL_ID = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"

PROMPT_TEMPLATES: Mapping[str, str] = {
    "answer_only": "{option}",
    "question_answer": "Question: {question} Answer: {option}",
    "biomedical_qa": (
        "This biomedical image answers the question '{question}' with '{option}'."
    ),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--project-root", type=Path, default=Path("."))
    p.add_argument("--data-root", type=Path, default=None)
    p.add_argument("--eval-csv", type=Path, required=True)
    p.add_argument("--image-root", type=Path, default=None)
    p.add_argument("--eval-limit", type=int, default=32)
    p.add_argument(
        "--frozen-template",
        choices=sorted(PROMPT_TEMPLATES),
        default=None,
        help="Use one template only. Set this for the final clean-test run.",
    )
    p.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    p.add_argument("--context-length", type=int, default=256)
    p.add_argument("--progress-every", type=int, default=10)
    p.add_argument("--out-dir", type=Path, required=True)
    return p.parse_args()


def choose_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise UserFacingError("CUDA was requested but is not available.")
    return torch.device(name)


def move_tokens(tokens: Any, device: torch.device) -> Any:
    if hasattr(tokens, "to"):
        return tokens.to(device)
    if isinstance(tokens, dict):
        return {k: v.to(device) for k, v in tokens.items()}
    raise TypeError(f"Unsupported tokenizer output type: {type(tokens)}")


def encode_text(model: Any, tokens: Any) -> torch.Tensor:
    if isinstance(tokens, dict):
        features = model.encode_text(**tokens)
    else:
        features = model.encode_text(tokens)
    return features / features.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def make_prompts(
    template_name: str, question: str, options: Sequence[str]
) -> List[str]:
    template = PROMPT_TEMPLATES[template_name]
    return [
        template.format(question=question.strip(), option=option.strip())
        for option in options
    ]


@torch.inference_mode()
def score_templates(
    model: Any,
    preprocess: Any,
    tokenizer: Any,
    device: torch.device,
    image: Image.Image,
    question: str,
    options: Sequence[str],
    template_names: Sequence[str],
    context_length: int,
) -> Dict[str, Dict[str, Any]]:
    image_tensor = preprocess(image.convert("RGB")).unsqueeze(0).to(device)

    use_amp = device.type == "cuda"
    with torch.autocast(
        device_type=device.type,
        dtype=torch.float16 if use_amp else torch.float32,
        enabled=use_amp,
    ):
        image_features = model.encode_image(image_tensor)
        image_features = image_features / image_features.norm(
            dim=-1, keepdim=True
        ).clamp_min(1e-12)

        outputs: Dict[str, Dict[str, Any]] = {}
        for template_name in template_names:
            prompts = make_prompts(template_name, question, options)
            tokens = tokenizer(prompts, context_length=context_length)
            tokens = move_tokens(tokens, device)
            text_features = encode_text(model, tokens)
            similarities = (image_features @ text_features.T).squeeze(0)
            pred_idx = int(torch.argmax(similarities).item())
            outputs[template_name] = {
                "pred_index": pred_idx,
                "pred_label": chr(ord("A") + pred_idx),
                "similarities": [float(x) for x in similarities.float().cpu()],
            }
    return outputs


def metric(correct: Sequence[bool]) -> Dict[str, float]:
    acc, low, high = bootstrap_ci_binary(correct)
    return {"accuracy": acc, "ci95_low": low, "ci95_high": high}


def main() -> int:
    args = parse_args()
    args.project_root = args.project_root.resolve()
    args.data_root = args.data_root.resolve() if args.data_root else None
    args.eval_csv = args.eval_csv.resolve()
    args.image_root = args.image_root.resolve() if args.image_root else None
    args.out_dir = args.out_dir.resolve()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.eval_limit <= 0:
        raise UserFacingError("--eval-limit must be positive.")

    try:
        from open_clip import create_model_from_pretrained, get_tokenizer
        import open_clip
    except ImportError as exc:
        raise UserFacingError(
            "open_clip is not installed. Run: "
            ".venv/Scripts/python.exe -m pip install open_clip_torch==2.23.0"
        ) from exc

    device = choose_device(args.device)
    log(f"device={device}")
    if device.type == "cuda":
        log(f"cuda_device={torch.cuda.get_device_name(0)}")

    template_names = (
        [args.frozen_template]
        if args.frozen_template
        else list(PROMPT_TEMPLATES)
    )

    log(f"loading_model={MODEL_ID}")
    model, preprocess = create_model_from_pretrained(MODEL_ID)
    tokenizer = get_tokenizer(MODEL_ID)
    model = model.to(device)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    log(f"total_parameters={total_params}")
    log(f"templates={template_names}")

    rows = read_csv_rows(args.eval_csv)[: args.eval_limit]
    image_roots = (
        [args.image_root]
        if args.image_root
        else discover_image_roots(args.project_root, args.data_root)
    )
    image_roots = [p for p in image_roots if p and p.is_dir()]
    if not image_roots:
        raise UserFacingError("No image root found. Pass --image-root.")
    image_index = index_images(image_roots)
    log(f"indexed_image_keys={len(image_index)}")

    correct_by_template: Dict[str, List[bool]] = {
        name: [] for name in template_names
    }
    predictions: List[Dict[str, Any]] = []
    skipped: Counter[str] = Counter()
    started = time.time()

    for row_index, row in enumerate(rows):
        try:
            question = get_question(row)
            options = clean_options(get_options(row))
            if len(options) < 2:
                raise UserFacingError("Fewer than two options.")
            gold = get_answer_label(row, options)
            if gold is None:
                raise UserFacingError("Could not resolve gold answer.")

            image_ref = get_image_ref(row)
            image_path = resolve_image_path(image_ref, image_index, image_roots)
            if image_path is None:
                skipped["missing_image"] += 1
                continue

            with Image.open(image_path) as raw:
                outputs = score_templates(
                    model=model,
                    preprocess=preprocess,
                    tokenizer=tokenizer,
                    device=device,
                    image=raw,
                    question=question,
                    options=options,
                    template_names=template_names,
                    context_length=args.context_length,
                )

            result: Dict[str, Any] = {
                "row_index": row_index,
                "image_ref": image_ref,
                "image_path": str(image_path),
                "question": question,
                "gold_label": gold,
            }
            for i, option in enumerate(options):
                result[f"choice_{chr(ord('A') + i)}"] = option

            for template_name, output in outputs.items():
                pred = output["pred_label"]
                is_correct = pred == gold
                correct_by_template[template_name].append(is_correct)
                result[f"{template_name}_pred_label"] = pred
                result[f"{template_name}_correct"] = int(is_correct)
                result[f"{template_name}_similarities_json"] = json.dumps(
                    output["similarities"]
                )

            predictions.append(result)
            done = len(predictions)
            if done == 1 or done % args.progress_every == 0:
                current = {
                    name: sum(vals) / len(vals)
                    for name, vals in correct_by_template.items()
                    if vals
                }
                best = max(current, key=current.get)
                log(
                    f"processed={done} elapsed_s={time.time()-started:.1f} "
                    f"best={best}:{current[best]:.4f}"
                )
        except Exception as exc:
            skipped[f"{type(exc).__name__}: {str(exc)[:120]}"] += 1

    if not predictions:
        raise UserFacingError(f"No samples evaluated. skipped={dict(skipped)}")

    metrics = {
        name: metric(correct)
        for name, correct in correct_by_template.items()
    }
    ranking = sorted(
        template_names,
        key=lambda name: (-metrics[name]["accuracy"], name),
    )
    elapsed = time.time() - started

    summary = {
        "run_name": args.out_dir.name,
        "model_id": MODEL_ID,
        "model_family": "BiomedCLIP",
        "evaluation_mode": "zero_shot_image_text_similarity",
        "eval_csv": str(args.eval_csv),
        "requested_samples": args.eval_limit,
        "evaluated_samples": len(predictions),
        "skipped": dict(skipped),
        "templates_evaluated": template_names,
        "prompt_templates": {
            name: PROMPT_TEMPLATES[name] for name in template_names
        },
        "frozen_template": args.frozen_template,
        "metrics": metrics,
        "validation_ranking": ranking,
        "best_validation_template": ranking[0],
        "total_parameters": total_params,
        "trainable_parameters_for_evaluation": 0,
        "evaluation_seconds": elapsed,
        "peak_gpu_memory_gb": (
            torch.cuda.max_memory_allocated() / (1024 ** 3)
            if device.type == "cuda"
            else 0.0
        ),
        "open_clip_version": getattr(open_clip, "__version__", "unknown"),
        "environment": env_summary(),
        "methodology_note": (
            "Prompt selection is allowed only on validation. Freeze the selected "
            "template before evaluating test_clean.csv."
        ),
    }

    write_csv_rows(args.out_dir / "biomedclip_predictions.csv", predictions)
    write_json(args.out_dir / "biomedclip_eval_summary.json", summary)

    for name in ranking:
        m = metrics[name]
        log(
            f"{name}: accuracy={m['accuracy']:.4f} "
            f"ci95=[{m['ci95_low']:.4f},{m['ci95_high']:.4f}]"
        )
    log(f"best_validation_template={ranking[0]}")
    log(f"evaluation_seconds={elapsed:.2f}")
    log(f"peak_gpu_memory_gb={summary['peak_gpu_memory_gb']:.3f}")
    log(f"summary={args.out_dir / 'biomedclip_eval_summary.json'}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        raise
