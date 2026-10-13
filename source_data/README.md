# Source data

Machine-readable derived data supporting the paper's main figures and supplementary analyses.

Main-figure source data:
- `figure2_canonical.csv`
- `figure3_grouped_cv.csv`
- `figure4_data_scaling.csv`
- `figure5_temporal.csv`

Additional analysis tables support the paired, grouped, scaling and temporal results reported in the manuscript and Supplementary Information.

Raw PMC-VQA question tables and source images are not redistributed here; see `../DATA_AND_LICENSE_NOTES.md`.


Revised-manuscript Stage 3 controls:
- `efficiency_audit_stage3.csv` — parameter counts, sequential inference timings, throughput, and peak PyTorch allocation on the fixed 1,000-question audit split.
- `image_shuffle_results_stage3.csv` — clean and per-seed image-shuffle accuracy/McNemar outputs.
- `image_shuffle_aggregate_stage3.csv` — mean and SD across the three deterministic image shuffles.

These Stage 3 files contain derived numerical outputs only; embedding caches and row-level image mappings are not redistributed.
