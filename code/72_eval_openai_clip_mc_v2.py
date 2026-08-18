#!/usr/bin/env python3
"""Zero-shot multiple-choice PMC-VQA evaluation with general OpenAI CLIP.\n\nCompatibility update: supports Transformers APIs where get_image_features\nand get_text_features return ModelOutput objects instead of raw tensors.

Protocol:
1. Compare the same three prompt templates used for BiomedCLIP on validation.
2. Freeze the best validation template.
3. Re-run on the canonical clean 2,000-sample test set.

The script removes positional option prefixes through common_vqa.clean_options,
uses no task-specific training, and audits text truncation at CLIP's fixed
context length (normally 77 tokens).
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

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

MODEL_ID = "openai/clip-vit-base-patch16"

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
        help="Use one already-selected template for the final clean-test run.",
    )
    p.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    p.add_argument("--progress-every", type=int, default=10)
    p.add_argument("--bootstrap-replicates", type=int, default=5000)
    p.add_argument("--local-files-only", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", type=Path, required=True)
    return p.parse_args()


def choose_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise UserFacingError("CUDA was requested but is not available.")
    return torch.device(name)


def make_prompts(
    template_name: str,
    question: str,
    options: Sequence[str],
) -> List[str]:
    template = PROMPT_TEMPLATES[template_name]
    return [
        template.format(question=question.strip(), option=option.strip())
        for option in options
    ]


def move_to_device(
    values: Mapping[str, torch.Tensor],
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in values.items()}


def extract_feature_tensor(output: Any) -> torch.Tensor:
    """Return projected CLIP features across Transformers API variants."""
    if isinstance(output, torch.Tensor):
        return output

    pooler_output = getattr(output, "pooler_output", None)
    if isinstance(pooler_output, torch.Tensor):
        return pooler_output

    image_embeds = getattr(output, "image_embeds", None)
    if isinstance(image_embeds, torch.Tensor):
        return image_embeds

    text_embeds = getattr(output, "text_embeds", None)
    if isinstance(text_embeds, torch.Tensor):
        return text_embeds

    if isinstance(output, (tuple, list)) and output:
        if isinstance(output[0], torch.Tensor):
            return output[0]

    raise TypeError(
        "Could not extract a feature tensor from "
        f"{type(output).__name__}."
    )


def normalize_features(output: Any) -> torch.Tensor:
    features = extract_feature_tensor(output)
    return features / features.norm(
        dim=-1,
        keepdim=True,
    ).clamp_min(1e-12)


def raw_token_lengths(tokenizer: Any, prompts: Sequence[str]) -> List[int]:
    encoded = tokenizer(
        list(prompts),
        add_special_tokens=True,
        truncation=False,
        padding=False,
        return_attention_mask=False,
    )
    return [len(ids) for ids in encoded["input_ids"]]


@torch.inference_mode()
def score_templates(
    model: Any,
    image_processor: Any,
    tokenizer: Any,
    device: torch.device,
    image: Image.Image,
    question: str,
    options: Sequence[str],
    template_names: Sequence[str],
    context_length: int,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    image_inputs = image_processor(
        images=image.convert("RGB"),
        return_tensors="pt",
    )
    image_inputs = move_to_device(image_inputs, device)

    use_amp = device.type == "cuda"
    amp_dtype = torch.float16 if use_amp else torch.float32

    with torch.autocast(
        device_type=device.type,
        dtype=amp_dtype,
        enabled=use_amp,
    ):
        image_features = normalize_features(
            model.get_image_features(**image_inputs)
        )

        outputs: Dict[str, Dict[str, Any]] = {}
        truncation: Dict[str, Dict[str, Any]] = {}

        for template_name in template_names:
            prompts = make_prompts(template_name, question, options)
            lengths = raw_token_lengths(tokenizer, prompts)
            truncated_flags = [length > context_length for length in lengths]

            text_inputs = tokenizer(
                prompts,
                padding=True,
                truncation=True,
                max_length=context_length,
                return_tensors="pt",
            )
            text_inputs = move_to_device(text_inputs, device)
            text_features = normalize_features(
                model.get_text_features(**text_inputs)
            )

            similarities = (image_features @ text_features.T).squeeze(0)
            pred_idx = int(torch.argmax(similarities).item())

            outputs[template_name] = {
                "pred_index": pred_idx,
                "pred_label": chr(ord("A") + pred_idx),
                "similarities": [
                    float(value)
                    for value in similarities.float().cpu()
                ],
            }
            truncation[template_name] = {
                "raw_token_lengths": lengths,
                "truncated_flags": truncated_flags,
                "truncated_prompt_count": sum(truncated_flags),
                "sample_had_truncation": any(truncated_flags),
            }

    return outputs, truncation


def bootstrap_ci_binary_custom(
    correct: Sequence[bool],
    replicates: int,
    seed: int,
) -> Tuple[float, float, float]:
    # Reuse the project's canonical point estimate and default CI whenever
    # possible. A custom seeded bootstrap is used when a replicate count is
    # explicitly supplied.
    if not correct:
        return 0.0, 0.0, 0.0

    accuracy = sum(correct) / len(correct)
    if replicates <= 0:
        canonical = bootstrap_ci_binary(correct)
        return canonical

    rng = random.Random(seed)
    n = len(correct)
    means: List[float] = []
    numeric = [1.0 if value else 0.0 for value in correct]

    for _ in range(replicates):
        means.append(
            sum(numeric[rng.randrange(n)] for _ in range(n)) / n
        )

    means.sort()
    low_index = max(0, int(0.025 * replicates))
    high_index = min(replicates - 1, int(0.975 * replicates))
    return accuracy, means[low_index], means[high_index]


def metric(
    correct: Sequence[bool],
    replicates: int,
    seed: int,
) -> Dict[str, float]:
    acc, low, high = bootstrap_ci_binary_custom(
        correct,
        replicates=replicates,
        seed=seed,
    )
    return {
        "accuracy": acc,
        "ci95_low": low,
        "ci95_high": high,
    }


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
    if not args.eval_csv.is_file():
        raise UserFacingError(f"Evaluation CSV not found: {args.eval_csv}")

    try:
        import transformers
        from transformers import AutoImageProcessor, AutoTokenizer, CLIPModel
    except ImportError as exc:
        raise UserFacingError(
            "transformers is not installed in the active environment."
        ) from exc

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = choose_device(args.device)
    log(f"device={device}")
    if device.type == "cuda":
        log(f"cuda_device={torch.cuda.get_device_name(0)}")
        torch.cuda.reset_peak_memory_stats()

    template_names = (
        [args.frozen_template]
        if args.frozen_template
        else list(PROMPT_TEMPLATES)
    )

    log(f"loading_model={MODEL_ID}")
    model = CLIPModel.from_pretrained(
        MODEL_ID,
        local_files_only=args.local_files_only,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        local_files_only=args.local_files_only,
    )
    image_processor = AutoImageProcessor.from_pretrained(
        MODEL_ID,
        local_files_only=args.local_files_only,
    )

    model = model.to(device)
    model.eval()

    context_length = int(
        getattr(model.config.text_config, "max_position_embeddings", 77)
    )
    total_params = sum(parameter.numel() for parameter in model.parameters())

    log(f"total_parameters={total_params}")
    log(f"context_length={context_length}")
    log(f"templates={template_names}")

    rows = read_csv_rows(args.eval_csv)[: args.eval_limit]
    image_roots = (
        [args.image_root]
        if args.image_root
        else discover_image_roots(args.project_root, args.data_root)
    )
    image_roots = [path for path in image_roots if path and path.is_dir()]
    if not image_roots:
        raise UserFacingError("No image root found. Pass --image-root.")

    image_index = index_images(image_roots)
    log(f"indexed_image_keys={len(image_index)}")

    correct_by_template: Dict[str, List[bool]] = {
        name: [] for name in template_names
    }
    truncation_prompt_counts: Counter[str] = Counter()
    truncation_sample_counts: Counter[str] = Counter()
    prompt_counts: Counter[str] = Counter()
    raw_length_sums: Counter[str] = Counter()
    raw_length_max: Counter[str] = Counter()

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
            image_path = resolve_image_path(
                image_ref,
                image_index,
                image_roots,
            )
            if image_path is None:
                skipped["missing_image"] += 1
                continue

            with Image.open(image_path) as raw_image:
                outputs, truncation = score_templates(
                    model=model,
                    image_processor=image_processor,
                    tokenizer=tokenizer,
                    device=device,
                    image=raw_image,
                    question=question,
                    options=options,
                    template_names=template_names,
                    context_length=context_length,
                )

            result: Dict[str, Any] = {
                "row_index": row_index,
                "image_ref": image_ref,
                "image_path": str(image_path),
                "question": question,
                "gold_label": gold,
            }

            for option_index, option in enumerate(options):
                option_label = chr(ord("A") + option_index)
                result[f"choice_{option_label}"] = option

            for template_name, output in outputs.items():
                prediction = output["pred_label"]
                is_correct = prediction == gold
                correct_by_template[template_name].append(is_correct)

                trunc = truncation[template_name]
                lengths = trunc["raw_token_lengths"]
                flags = trunc["truncated_flags"]

                prompt_counts[template_name] += len(lengths)
                truncation_prompt_counts[template_name] += sum(flags)
                truncation_sample_counts[template_name] += int(any(flags))
                raw_length_sums[template_name] += sum(lengths)
                raw_length_max[template_name] = max(
                    raw_length_max[template_name],
                    max(lengths) if lengths else 0,
                )

                result[f"{template_name}_pred_label"] = prediction
                result[f"{template_name}_correct"] = int(is_correct)
                result[f"{template_name}_similarities_json"] = json.dumps(
                    output["similarities"]
                )
                result[f"{template_name}_raw_token_lengths_json"] = json.dumps(
                    lengths
                )
                result[f"{template_name}_truncated_flags_json"] = json.dumps(
                    flags
                )
                result[f"{template_name}_sample_had_truncation"] = int(
                    any(flags)
                )

            predictions.append(result)
            done = len(predictions)

            if done == 1 or done % args.progress_every == 0:
                current = {
                    name: sum(values) / len(values)
                    for name, values in correct_by_template.items()
                    if values
                }
                best = max(current, key=current.get)
                log(
                    f"processed={done} "
                    f"elapsed_s={time.time() - started:.1f} "
                    f"best={best}:{current[best]:.4f}"
                )

        except Exception as exc:
            key = f"{type(exc).__name__}: {str(exc)[:160]}"
            skipped[key] += 1

    if not predictions:
        raise UserFacingError(
            f"No samples evaluated. skipped={dict(skipped)}"
        )

    metrics = {
        name: metric(
            correct,
            replicates=args.bootstrap_replicates,
            seed=args.seed,
        )
        for name, correct in correct_by_template.items()
    }
    ranking = sorted(
        template_names,
        key=lambda name: (-metrics[name]["accuracy"], name),
    )
    elapsed = time.time() - started

    truncation_summary: Dict[str, Dict[str, Any]] = {}
    evaluated_samples = len(predictions)

    for name in template_names:
        total_prompts = prompt_counts[name]
        truncated_prompts = truncation_prompt_counts[name]
        truncated_samples = truncation_sample_counts[name]

        truncation_summary[name] = {
            "context_length": context_length,
            "total_prompts": total_prompts,
            "truncated_prompts": truncated_prompts,
            "truncated_prompt_rate": (
                truncated_prompts / total_prompts
                if total_prompts
                else 0.0
            ),
            "samples_with_any_truncation": truncated_samples,
            "sample_truncation_rate": (
                truncated_samples / evaluated_samples
                if evaluated_samples
                else 0.0
            ),
            "mean_raw_token_length": (
                raw_length_sums[name] / total_prompts
                if total_prompts
                else 0.0
            ),
            "max_raw_token_length": raw_length_max[name],
        }

    summary = {
        "run_name": args.out_dir.name,
        "model_id": MODEL_ID,
        "model_family": "general_OpenAI_CLIP",
        "visual_backbone": "ViT-B/16",
        "evaluation_mode": "zero_shot_image_text_similarity",
        "eval_csv": str(args.eval_csv),
        "image_roots": [str(path) for path in image_roots],
        "requested_samples": args.eval_limit,
        "evaluated_samples": len(predictions),
        "skipped": dict(skipped),
        "templates_evaluated": template_names,
        "prompt_templates": {
            name: PROMPT_TEMPLATES[name]
            for name in template_names
        },
        "frozen_template": args.frozen_template,
        "metrics": metrics,
        "validation_ranking": ranking,
        "best_validation_template": ranking[0],
        "context_length": context_length,
        "truncation_audit": truncation_summary,
        "total_parameters": total_params,
        "trainable_parameters_for_evaluation": 0,
        "evaluation_seconds": elapsed,
        "peak_gpu_memory_gb": (
            torch.cuda.max_memory_allocated() / (1024 ** 3)
            if device.type == "cuda"
            else 0.0
        ),
        "transformers_version": getattr(
            transformers,
            "__version__",
            "unknown",
        ),
        "torch_version": torch.__version__,
        "environment": env_summary(),
        "methodology_note": (
            "Prompt templates match the BiomedCLIP validation protocol. "
            "Options are prefix-free. The best validation template must be "
            "frozen before the clean-test run."
        ),
    }

    write_csv_rows(
        args.out_dir / "clip_predictions.csv",
        predictions,
    )
    write_json(
        args.out_dir / "clip_eval_summary.json",
        summary,
    )

    log(f"evaluated_samples={len(predictions)}")
    for name in ranking:
        item = metrics[name]
        trunc = truncation_summary[name]
        log(
            f"{name}: accuracy={item['accuracy']:.4f} "
            f"ci95=[{item['ci95_low']:.4f},{item['ci95_high']:.4f}] "
            f"truncated_prompts={trunc['truncated_prompts']}/"
            f"{trunc['total_prompts']} "
            f"truncated_samples={trunc['samples_with_any_truncation']}/"
            f"{len(predictions)}"
        )

    log(f"best_validation_template={ranking[0]}")
    log(f"evaluation_seconds={elapsed:.2f}")
    log(f"peak_gpu_memory_gb={summary['peak_gpu_memory_gb']:.3f}")
    log(f"summary={args.out_dir / 'clip_eval_summary.json'}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        raise SystemExit(130)
