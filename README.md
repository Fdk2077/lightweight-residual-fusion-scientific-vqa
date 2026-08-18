# Lightweight Residual Fusion for Scientific VQA

Reproducibility repository for:

**Lightweight residual fusion reveals temporal generalization limits in scientific visual question answering**

Authors: **Farshad Kazemi** and **Keivan Borna**  
Affiliation: Faculty of Mathematical Sciences and Computer, Kharazmi University, Tehran, Iran

## Overview

This repository contains the custom code and aggregate source data used to reproduce the main analyses in the manuscript. The study evaluates frozen CLIP and BiomedCLIP baselines, limited BLIP LoRA adaptation, and an option-shared residual scorer over frozen general and biomedical image-text representations on PMC-VQA.

The main methodological contribution is an option-shared residual scorer that applies the same correction function to every answer option, preserving exact option-permutation equivariance while training only a small adaptation head.

## Repository contents

- `code/` — training, evaluation, ablation, grouped cross-validation, data-scaling, and temporal-audit scripts.
- `source_data/` — aggregate/derived numerical data underlying the main quantitative figures and tables.
- `docs/EXPERIMENT_REGISTRY_V10.csv` — experiment registry.
- `REPRODUCIBILITY_MANIFEST.md` — mapping between manuscript analyses, scripts, and source-data files.
- `DATA_AND_LICENSE_NOTES.md` — data provenance and redistribution boundaries.
- `environment_archived.json` and `environment_known.txt` — archived environment information.
- `requirements_reference.txt` — reference Python dependencies from the archived study environment.

## Raw data

Raw PMC-VQA tables and the source-image archive are **not redistributed here**. Obtain PMC-VQA from the original dataset providers and comply with the dataset and underlying article licenses.

The public repository intentionally excludes row-level temporal audit files containing PMC-VQA-derived question/answer text. The included scripts reconstruct the temporal audit from locally obtained PMC-VQA resources and NCBI metadata.

## Principal model identifiers

- BLIP: `Salesforce/blip-vqa-base`
- CLIP: `openai/clip-vit-base-patch16`
- BiomedCLIP: `microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224`

## Main reproduction order

1. Frozen baseline evaluation / prompt protocol:
   - `code/64_eval_biomedclip_mc_v1.py`
   - `code/72_eval_openai_clip_mc_v2.py`
   - BLIP utilities in `code/common_vqa.py`
2. BLIP LoRA:
   - `code/train_blip_lora_pmc_v2.py`
   - `code/61_train_blip_cross_attention_lora_v3.py`
3. Canonical baseline comparison:
   - `code/73_compare_blip_clip_biomedclip_paired_v1.py`
4. Residual scorer training:
   - `code/79_train_joint_clip_biomed_residual_v1.py`
5. Prespecified gate:
   - `code/80_eval_joint_clip_biomed_holdout_v1.py`
6. One-shot canonical evaluation:
   - `code/81_eval_joint_clip_biomed_test_v1.py`
7. Feature ablations:
   - `code/run_paper_e1_feature_ablations_v1.py`
8. PMCID-grouped cross-validation:
   - `code/run_paper_e1c_group_cv_v1.py`
9. Leakage-free data scaling:
   - `code/run_paper_e2_data_scaling_v1_ready.py`
10. Post-cutoff temporal audit:
   - `code/run_paper_e4a_temporal_candidate_audit_v3.py`
11. Temporal evaluation:
   - `code/run_paper_e4b_temporal_evaluation_v2.py`

## Random seeds

Where model training uses multiple seeds, the fixed seeds are:

`42`, `7`, `123`

## Hardware context

The experiments were designed to run under a constrained local setup centered on an NVIDIA GTX 1650 with 4 GB VRAM. See the manuscript and archived environment files for details.

## NCBI metadata access

The temporal-audit script queries NCBI PMC metadata. It can run without an API key at a lower request rate. If a key is used, it can be supplied through the `NCBI_API_KEY` environment variable or at runtime. **No API keys or credentials are included in this repository.**

## Reproducibility notes

The repository contains the numerical source data underlying the main figures and key analyses. Some experiments depend on locally generated frozen feature caches and raw PMC-VQA resources that are not redistributed.

One dependency used for BiomedCLIP (`open_clip_torch`) was not version-recorded in the archived environment report. This uncertainty is disclosed rather than guessed.

## Citation

A `CITATION.cff` file is included for repository citation. The journal citation/DOI should be added after publication.

## License

The custom software in this repository is released under the **MIT License**. See `LICENSE`.

This license applies only to the authors' custom code in this repository. PMC-VQA data, source images, pretrained model weights, and other third-party materials remain governed by their original licenses and terms.

The associated journal article is licensed separately by the publisher. For Communications Medicine, the intended article license is **CC BY 4.0**, subject to the journal's publication process and final author selection.

## Contact

- Farshad Kazemi: `farshadbb23@gmail.com`
- Keivan Borna: `borna@khu.ac.ir`
