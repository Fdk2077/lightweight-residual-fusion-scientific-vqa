# Lightweight Residual Fusion for Scientific VQA

[![DOI](https://zenodo.org/badge/1338744814.svg)](https://doi.org/10.5281/zenodo.22045938)

Reproducibility repository for:

**Article-grouped and post-cutoff evaluation reveals limits of parameter-efficient adaptation in scientific visual question answering**

Authors: **Farshad Kazemi** and **Keivan Borna**  
Affiliation: Faculty of Mathematical Sciences and Computer, Kharazmi University, Tehran, Iran

## Overview

This repository contains the custom code and aggregate source data used to reproduce the main analyses in the manuscript. The study evaluates frozen CLIP and BiomedCLIP baselines, limited BLIP LoRA adaptation, and an option-shared residual scorer over frozen general and biomedical image-text representations on PMC-VQA.

The revised study treats the option-shared residual scorer as a controlled parameter-efficient adaptation mechanism rather than as an end-to-end lightweight model. The shared scoring function preserves exact option-permutation equivariance while the evaluation emphasizes article grouping, temporal generalization, model-complexity controls, and post-hoc image-correspondence/efficiency checks.

Repository: https://github.com/Fdk2077/lightweight-residual-fusion-scientific-vqa

Version v1.1.0 DOI: https://doi.org/10.5281/zenodo.23376674

Zenodo concept DOI (all versions/latest): https://doi.org/10.5281/zenodo.22045938

## Repository contents

- `code/` — training, evaluation, ablation, grouped cross-validation, data-scaling, temporal-audit, efficiency-audit, and image-correspondence-control scripts.
- `source_data/` — aggregate/derived numerical data underlying the main quantitative figures and tables.
- `docs/EXPERIMENT_REGISTRY.csv` — experiment registry.
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
   - `code/64_eval_biomedclip_mc.py`
   - `code/72_eval_openai_clip_mc.py`
   - BLIP utilities in `code/common_vqa.py`
2. BLIP LoRA:
   - `code/train_blip_lora_pmc.py`
   - `code/61_train_blip_cross_attention_lora.py`
3. Canonical baseline comparison:
   - `code/73_compare_blip_clip_biomedclip_paired.py`
4. Residual scorer training:
   - `code/79_train_joint_clip_biomed_residual.py`
5. Prespecified gate:
   - `code/80_eval_joint_clip_biomed_holdout.py`
6. One-shot canonical evaluation:
   - `code/81_eval_joint_clip_biomed_test.py`
7. Feature ablations:
   - `code/run_paper_e1_feature_ablations.py`
8. PMCID-grouped cross-validation:
   - `code/run_paper_e1c_group_cv.py`
9. Leakage-free data scaling:
   - `code/run_paper_e2_data_scaling.py`
10. Post-cutoff temporal audit:
   - `code/run_paper_e4a_temporal_candidate_audit.py`
11. Temporal evaluation:
   - `code/run_paper_e4b_temporal_evaluation.py`
12. Post-hoc efficiency audit:
   - `code/run_efficiency_audit_v1.py`
13. Post-hoc image-correspondence control:
   - `code/run_image_shuffle_control_v4_precision_fixed.py`

## Random seeds

Where model training uses multiple seeds, the fixed seeds are:

`42`, `7`, `123`

## Hardware context

The experiments were designed to run under a constrained local setup centered on an NVIDIA GTX 1650 with 4 GB VRAM. The revised efficiency audit shows that the adaptation head is parameter-efficient, while the complete dual-encoder inference stack is much larger; see `source_data/efficiency_audit_stage3.csv` and the manuscript for details.

## NCBI metadata access

The temporal-audit script queries NCBI PMC metadata. It can run without an API key at a lower request rate. If a key is used, it can be supplied through the `NCBI_API_KEY` environment variable or at runtime. **No API keys or credentials are included in this repository.**

## Reproducibility notes

The repository contains the numerical source data underlying the main figures and key analyses. Some experiments depend on locally generated frozen feature caches and raw PMC-VQA resources that are not redistributed.

One dependency used for BiomedCLIP (`open_clip_torch`) was not version-recorded in the archived environment report. This uncertainty is disclosed rather than guessed.

## Citation

For the immutable software snapshot used with the revised manuscript, cite DOI `10.5281/zenodo.23376674` (version `v1.1.0`). The DOI badge at the top points to the Zenodo concept record (`10.5281/zenodo.22045938`), which resolves to the latest archived version.

A `CITATION.cff` file is included for machine-readable repository citation. The journal article citation/DOI should be added after publication.

## License

The custom software in this repository is released under the **MIT License**. See `LICENSE`.

This license applies only to the authors' custom code in this repository. PMC-VQA data, source images, pretrained model weights, and other third-party materials remain governed by their original licenses and terms.

The associated journal article is licensed separately under the terms selected with the eventual publisher.

## Contact

- Farshad Kazemi: `farshadbb23@gmail.com`
- Keivan Borna: `borna@khu.ac.ir`

## Revised Stage 3 controls

The revised manuscript adds two post-hoc analyses that do not retrain or select models:

- an end-to-end efficiency audit on the fixed 1,000-question development split;
- a deterministic cross-PMCID image-correspondence shuffle control on the canonical 2,000-question test, using seeds 42, 7, and 123.

The public repository includes the executed scripts and derived numerical source data, but excludes large embedding caches and row-level shuffle mappings.

## Release integrity

The `v1.1.0` Zenodo snapshot is archived under DOI `10.5281/zenodo.23376674`. `SHA256SUMS.txt` records hashes for the files included in that tagged release.
