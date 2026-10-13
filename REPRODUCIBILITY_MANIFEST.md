# Reproducibility manifest

| Manuscript item / analysis | Primary source-data file(s) | Primary script(s) |
|---|---|---|
| Canonical frozen/model accuracy (Fig. 2, Table 2) | `source_data/figure2_canonical.csv` | `code/73_compare_blip_clip_biomedclip_paired.py`, `code/81_eval_joint_clip_biomed_test.py` |
| Gate paired comparison (Table 3) | `source_data/pairwise_gate_ablations.csv` | `code/80_eval_joint_clip_biomed_holdout.py`, `code/run_paper_e1_feature_ablations.py` |
| Gate residual ablations (Table 4; Supplementary Fig. 1/Table 1) | `source_data/table_gate_ablations.csv`, `source_data/pairwise_gate_ablations.csv` | `code/run_paper_e1_feature_ablations.py` |
| PMCID-grouped CV (Fig. 3; Supplementary Table 2/Data 2) | `source_data/figure3_grouped_cv.csv`, `source_data/grouped_cv_fold_results.csv`, `source_data/grouped_cv_cluster_comparisons.csv` | `code/run_paper_e1c_group_cv.py` |
| Leakage-free scaling (Fig. 4; Supplementary Table 3/Data 3) | `source_data/figure4_data_scaling.csv`, `source_data/data_scaling_main.csv`, `source_data/data_scaling_seed_stability.csv`, `source_data/data_scaling_full_vs_linear_cluster.csv` | `code/run_paper_e2_data_scaling.py` |
| Temporal model accuracy (Fig. 5, Table 5) | `source_data/figure5_temporal.csv`, `source_data/temporal_main.csv` | `code/run_paper_e4a_temporal_candidate_audit.py`, `code/run_paper_e4b_temporal_evaluation.py` |
| Temporal pairwise comparisons (Supplementary Table 4) | `source_data/temporal_main.csv` | `code/run_paper_e4b_temporal_evaluation.py` |
| Exploratory temporal subgroups (Supplementary Data 1) | `source_data/temporal_exploratory_subgroups.csv` | `code/run_paper_e4b_temporal_evaluation.py` |
| Temporal quality exclusions | Row-level audit outputs intentionally excluded from the public repository; reconstruct with locally obtained PMC-VQA data | `code/run_paper_e4a_temporal_candidate_audit.py`, `code/run_paper_e4b_temporal_evaluation.py` |
| BLIP standard LoRA | experiment outputs referenced in registry | `code/train_blip_lora_pmc.py` |
| BLIP cross-attention LoRA | experiment outputs referenced in registry | `code/61_train_blip_cross_attention_lora.py` |
| Post-hoc efficiency audit | `source_data/efficiency_audit_stage3.csv` | `code/run_efficiency_audit_v1.py` |
| Post-hoc image-correspondence control (Table 6) | `source_data/image_shuffle_results_stage3.csv`, `source_data/image_shuffle_aggregate_stage3.csv` | `code/run_image_shuffle_control_v4_precision_fixed.py` |

The paper-stage experiment registry is `docs/EXPERIMENT_REGISTRY.csv`.

Stage 3 controls are post-hoc sensitivity/efficiency analyses. They perform no model training or model selection and must not be interpreted as reopening the canonical test.
