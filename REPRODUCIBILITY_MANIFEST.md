# Reproducibility manifest

| Manuscript item / analysis | Primary source-data file(s) | Primary script(s) |
|---|---|---|
| Canonical frozen/model accuracy (Fig. 2, Table 2) | `source_data/figure2_canonical.csv` | `code/73_compare_blip_clip_biomedclip_paired_v1.py`, `code/81_eval_joint_clip_biomed_test_v1.py` |
| Gate paired comparison (Table 3) | `source_data/pairwise_gate_ablations.csv` | `code/80_eval_joint_clip_biomed_holdout_v1.py`, `code/run_paper_e1_feature_ablations_v1.py` |
| Gate residual ablations (Table 4; Supplementary Fig. 1/Table 1) | `source_data/table_gate_ablations.csv`, `source_data/pairwise_gate_ablations.csv` | `code/run_paper_e1_feature_ablations_v1.py` |
| PMCID-grouped CV (Fig. 3; Supplementary Table 2/Data 2) | `source_data/figure3_grouped_cv.csv`, `source_data/grouped_cv_fold_results.csv`, `source_data/grouped_cv_cluster_comparisons.csv` | `code/run_paper_e1c_group_cv_v1.py` |
| Leakage-free scaling (Fig. 4; Supplementary Table 3/Data 3) | `source_data/figure4_data_scaling.csv`, `source_data/data_scaling_main.csv`, `source_data/data_scaling_seed_stability.csv`, `source_data/data_scaling_full_vs_linear_cluster.csv` | `code/run_paper_e2_data_scaling_v1_ready.py` |
| Temporal model accuracy (Fig. 5, Table 5) | `source_data/figure5_temporal.csv`, `source_data/temporal_main.csv` | `code/run_paper_e4a_temporal_candidate_audit_v3.py`, `code/run_paper_e4b_temporal_evaluation_v2.py` |
| Temporal pairwise comparisons (Supplementary Table 4) | `source_data/temporal_main.csv` | `code/run_paper_e4b_temporal_evaluation_v2.py` |
| Exploratory temporal subgroups (Supplementary Data 1) | `source_data/temporal_exploratory_subgroups.csv` | `code/run_paper_e4b_temporal_evaluation_v2.py` |
| Temporal quality exclusions | Row-level audit outputs intentionally excluded from the public repository; reconstruct with the included scripts and locally obtained PMC-VQA data | `code/run_paper_e4a_temporal_candidate_audit_v3.py`, `code/run_paper_e4b_temporal_evaluation_v2.py` |
| BLIP standard LoRA | experiment outputs referenced in registry | `code/train_blip_lora_pmc_v2.py` |
| BLIP cross-attention LoRA | experiment outputs referenced in registry | `code/61_train_blip_cross_attention_lora_v3.py` |

The experiment registry `docs/EXPERIMENT_REGISTRY_V10.csv` provides the paper-stage record of the completed experiments.
