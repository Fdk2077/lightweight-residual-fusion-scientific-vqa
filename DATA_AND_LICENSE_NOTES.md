# Data and license notes

## Raw benchmark data

The study reuses PMC-VQA, a publicly released benchmark derived from PubMed Central scientific figures. Raw PMC-VQA tables and the full source-image archive are not redistributed in this public repository. Users should obtain them from the original dataset providers and comply with the dataset and underlying article licenses.

## Public source data

The CSV files under `source_data/` contain derived aggregate/model-evaluation outputs used to produce the paper figures, tables, and supplementary analyses.

## Row-level temporal audit data

The internal reproducibility package contains row-level temporal quality-control material derived from PMC-VQA. Those files include benchmark question/answer content and are intentionally **excluded from this public repository** pending any redistribution analysis under the original dataset terms.

The temporal selection and evaluation logic is fully represented by:

- `code/run_paper_e4a_temporal_candidate_audit.py`
- `code/run_paper_e4b_temporal_evaluation.py`

and can be rerun by users who obtain PMC-VQA through the original distribution channel.

## Qualitative source images

Qualitative examples used in the manuscript originate from PubMed Central source articles. The raw source images are not redistributed in this repository. Any reuse remains subject to the license of the corresponding source article.

## Third-party models

This repository does not redistribute pretrained model weights. Users should obtain BLIP, CLIP, and BiomedCLIP from their original providers and follow the corresponding licenses and terms.


## Stage 3 controls

The revised manuscript adds an efficiency audit and an image-correspondence shuffle control. Public source data include only aggregate/per-shuffle numerical outputs; embedding caches, raw images, local absolute paths, and row-level question/answer text are not redistributed.

The corresponding scripts are:

- `code/run_efficiency_audit_v1.py`
- `code/run_image_shuffle_control_v4_precision_fixed.py`
