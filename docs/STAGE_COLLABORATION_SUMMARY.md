# STAMP Project Stage Collaboration Summary

Last updated: 2026-05-18

## 1. Document purpose

This document is a stage handoff summary for collaborators joining the project now.
It records:

- What has been attempted from the beginning of this project to the current point
- Which source files were added or modified
- Which experiment lines are already validated
- What the current best results are
- What problems remain unsolved
- Where a new collaborator should start

The current primary target is still:

- Internal AUROC > 0.8
- External TCGA AUROC > 0.8

This target has not yet been reached.

## 2. Problem setup

Task:

- Binary classification of early recurrence after HCC surgery
- Positive class: `Early recurrence = 1`

Datasets used in this stage:

- Internal cohort: `ourdata`
- External cohort: `TCGA`

High-level practical observation so far:

- Internal performance can be improved by adding clinical information
- External TCGA performance is much harder to improve
- The main bottleneck is still domain shift, not just calibration

## 3. Repository areas changed in this stage

### 3.1 Core modeling and training

Main modified files:

- `src/stamp/modeling/config.py`
- `src/stamp/modeling/data.py`
- `src/stamp/modeling/train.py`
- `src/stamp/modeling/crossval.py`
- `src/stamp/modeling/registry.py`
- `src/stamp/modeling/models/__init__.py`
- `src/stamp/modeling/models/abmil.py`
- `src/stamp/modeling/models/trans_mil.py`
- `src/stamp/config.yaml`

Main changes:

- Added domain adaptation related config fields:
  - `use_coral`
  - `coral_weight`
  - `coral_warmup_epochs`
  - `use_dann`
  - `dann_weight`
  - `dann_warmup_epochs`
- Added target-domain training inputs:
  - `target_feature_dir`
  - `target_slide_table`
  - `target_clini_table`
  - `target_pseudolabel_csv`
- Added pseudo-label and distillation related config fields:
  - `use_pseudolabels`
  - `pseudolabel_loss_weight`
  - `pseudolabel_warmup_epochs`
  - `pseudolabel_confidence_threshold`
  - `use_distillation`
  - `distillation_loss_weight`
  - `distillation_warmup_epochs`
  - `distillation_confidence_threshold`
  - `distillation_temperature`
- Extended cross-validation and training loops so that source and target dataloaders can be used together
- Added DANN domain classifier and related training logic in the Lightning model
- Added CORAL alignment loss support
- Added pseudo-label loss and soft distillation loss support

### 3.2 Clinical feature integration

Added files:

- `src/stamp/modeling/clinical.py`
- `src/stamp/modeling/models/abmil_clinical.py`

Modified files:

- `src/stamp/modeling/data.py`
- `src/stamp/modeling/train.py`
- `src/stamp/modeling/registry.py`
- `src/stamp/modeling/config.py`
- `src/stamp/modeling/models/__init__.py`

Main changes:

- Added clinical feature normalization and transformation utilities
- Extended `PatientData` to carry `clinical_features`
- Added multimodal bag batches of the form:
  - `(bags, coords, bag_sizes, targets, clinical_tensor)`
- Added `abmil_clinical` model and registered it in the model registry
- Enabled end-to-end pathology + clinical joint training

### 3.3 Utility and experiment scripts

Added or heavily modified scripts:

- `scripts/clinical_ensemble.py`
- `scripts/clinical_transfer_ensemble.py`
- `scripts/clinical_transfer_stack.py`
- `scripts/model_ensemble.py`
- `scripts/macenko_normalize_tiles.py`
- `scripts/test_time_feature_align.py`

Purpose of each:

- `clinical_ensemble.py`
  - Post-hoc fusion of WSI predictions and internal clinical model
- `clinical_transfer_ensemble.py`
  - Apply internal clinical model idea to TCGA using shared clinical fields
- `clinical_transfer_stack.py`
  - External stacking transfer experiment on shared clinical features
- `model_ensemble.py`
  - Post-hoc probability ensemble for crossval and deploy outputs
  - Later extended to support deploy-time weight search
- `macenko_normalize_tiles.py`
  - Apply Macenko stain normalization to cached tiles before feature extraction
- `test_time_feature_align.py`
  - Test-time feature alignment / TTA on target-domain bags
  - Later extended with temperature scaling and checkpoint-weighted integration

## 4. Important bugs fixed during this stage

These are worth knowing because they may recur if new code is added around the same areas.

### 4.1 Target dataloader missing during domain adaptation

Symptom:

- `use_coral=True but no target-domain dataloader was provided`

Fix area:

- `src/stamp/modeling/crossval.py`
- `src/stamp/modeling/train.py`
- `src/stamp/modeling/models/__init__.py`

Meaning:

- Domain adaptation configs now require correctly built target-domain dataloaders.

### 4.2 CPU / GPU device mismatch

Symptom:

- `Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu`

Fix area:

- `src/stamp/modeling/models/__init__.py`
- `scripts/test_time_feature_align.py`

Meaning:

- Target-domain inputs and clinical tensors must be explicitly moved to the model device.

### 4.3 Clinical batch shape mismatch

Symptom:

- `too many values to unpack (expected 4)`
- `_collate_to_tuple()` received 5-item batches

Root cause:

- `tile_bag_dataloader()` passed a list of `None` clinical features into `BagDataset`, causing 5-item outputs even for pathology-only workflows.

Fix area:

- `src/stamp/modeling/data.py`

Meaning:

- Clinical tensors are now only passed when at least one patient actually has clinical features.

### 4.4 Macenko normalization failure on tile shape handling

Symptom:

- `TypeError: Cannot handle this data type: (1, 1, 224), |u1`

Fix area:

- `scripts/macenko_normalize_tiles.py`

Meaning:

- Debugged normalization backend and tile conversion logic so full-cache normalization could complete with only a small number of fallback copies.

### 4.5 Statistics config format mistake

Symptom:

- `Extra inputs are not permitted` when running `uv run stamp -c ... statistics`

Meaning:

- Statistics config files must use top-level key `statistics:`

This is a usage note rather than a source-code bug, but it caused repeated confusion and is now worth documenting for collaborators.

### 4.6 Torch / CUDA shared library issue on server

Symptom:

- `undefined symbol: __nvJitLinkComplete_12_4`

Practical workaround used:

- `export LD_LIBRARY_PATH=""`

This was a runtime environment issue and was not solved by repository code changes.

## 5. Experiment chronology and results

The tables below summarize the major experiment families that were executed and kept as reference.

### 5.1 Early pathology baselines and classic ensembles

| Experiment | Description | Internal AUROC | External TCGA AUROC | Notes |
| --- | --- | ---: | ---: | --- |
| `exp01_abmil` | ABMIL baseline | 0.7443 | - | Early strong internal baseline |
| `exp07_uni2` | UNI2 features | 0.7517 | - | Good internal only |
| `exp10_phikon` | Phikon features | 0.7511 | - | Good internal only |
| `exp12_phikon_subbag` | Phikon subbag | 0.7541 | 0.5140 | Weak TCGA transfer |
| `exp11_abmil_subbag_tcga` | ABMIL subbag deploy | - | 0.5757 | Better external than Phikon subbag |
| `exp13_conch_phikon_ensemble` | `exp01_abmil` + `exp12_phikon_subbag` | 0.7605 | 0.5315 | Internal gain, external drop |
| `exp14_uni2_subbag` | UNI2 subbag | 0.7612 | 0.5079 | Highest early internal among pathology-only, poor TCGA |

Takeaway:

- Pure model ensembling improved internal AUROC more easily than external AUROC.
- External TCGA transfer remained weak in this phase.

### 5.2 Clinical post-hoc fusion and clinical transfer

| Experiment | Description | Internal AUROC | External TCGA AUROC | Notes |
| --- | --- | ---: | ---: | --- |
| `exp01_abmil_clinpath` | Internal post-hoc WSI + clinical ensemble | 0.8394 | - | Best internal result so far |
| `exp01_abmil_clinpath_tcga` | Transfer post-hoc clinical fusion to TCGA | - | 0.5248 | Did not help TCGA |
| `exp01_abmil_clinstack_tcga` | Clinical transfer stacking to TCGA | - | 0.5510 | Better than simple transfer ensemble, still weak |

Takeaway:

- Clinical information is useful for internal prediction.
- The same clinical fusion idea does not transfer well to TCGA.

### 5.3 Macenko normalization and stain handling

| Experiment | Description | Internal AUROC | External TCGA AUROC | Notes |
| --- | --- | ---: | ---: | --- |
| `exp16_macenko_conch_subbag_tcga` | TCGA Macenko-normalized features | - | 0.6026 | Better than several early external baselines |
| `exp17_macenko_tcga_ref` | Macenko using TCGA-style reference | 0.6702 | 0.5549 | Internal degraded, external still limited |

Operational note:

- Macenko normalization eventually completed successfully with a small number of fallback tile copies.
- This line improved stain consistency but did not solve the full domain-shift problem.

### 5.4 CORAL and DANN domain adaptation

| Experiment | Description | Internal AUROC | External TCGA AUROC | Notes |
| --- | --- | ---: | ---: | --- |
| `exp18_abmil_coral` | ABMIL + CORAL | 0.7165 | 0.6162 | Helpful externally vs some early baselines |
| `exp18b_abmil_coral_w020` | ABMIL + CORAL weight 0.20 | 0.7037 | 0.6003 | Worse than default CORAL |
| `exp18c_abmil_coral_w050` | ABMIL + CORAL weight 0.50 | 0.7034 | 0.6077 | Slightly better than w020 |
| `exp19_transmil_coral_w020` | TransMIL + CORAL | 0.7334 | 0.5837 | External weaker than ABMIL CORAL |
| `exp20_abmil_dann` | ABMIL + DANN | 0.7207 | 0.6224 | Better than CORAL |
| `exp20b_abmil_dann_w050` | ABMIL + DANN weight 0.50 | 0.7206 | 0.6302 | Best external training-time model so far |
| `exp20c_abmil_dann_w020_long` | ABMIL + DANN long training | 0.7219 | 0.6138 | No external gain |
| `exp20d_abmil_dann_w080` | ABMIL + DANN weight 0.80 | 0.7173 | 0.6298 | Very close to `exp20b` |
| `exp20e_abmil_dann_w100` | ABMIL + DANN weight 1.00 | 0.7181 | 0.6296 | Very close to `exp20b` |
| `exp20f_abmil_dann_w050_bigdomain` | Larger target-domain setting | 0.7034 | 0.6225 | No gain |
| `exp21_transmil_dann_w020` | TransMIL + DANN | 0.7184 | 0.6061 | Worse than ABMIL DANN |

Takeaway:

- DANN was the most useful training-time domain adaptation family in this stage.
- `exp20b_abmil_dann_w050` became the main pathology-only external reference point.

### 5.5 Pseudo-label and distillation lines

| Experiment | Description | Internal AUROC | External TCGA AUROC | Notes |
| --- | --- | ---: | ---: | --- |
| `exp22_abmil_pseudolabel` | Hard pseudo-label training | 0.7196 | 0.5875 | Did not help external transfer |
| `exp23_abmil_distill` | Soft-label distillation | 0.7099 | 0.6233 | Better than hard pseudo-label, below `exp20b` |
| `exp24_abmil_dann_distill` | DANN + distillation | 0.7064 | 0.6106 | Joint strategy underperformed |
| `exp25_abmil_dann_distill_ensemble` | Ensemble of DANN and distillation lines | 0.7173 | 0.6287 | Very close to `exp20b`, not better |

Takeaway:

- Distillation was better than hard pseudo-labeling.
- None of these variants surpassed `exp20b` on TCGA.

### 5.6 End-to-end pathology + clinical models

| Experiment | Description | Internal AUROC | External TCGA AUROC | Notes |
| --- | --- | ---: | ---: | --- |
| `exp26_abmil_clinical_joint` | End-to-end pathology + clinical | 0.7471 | 0.6094 | Internal gain, external drop |
| `exp27_abmil_clinical_dann` | Multimodal pathology + clinical + DANN | 0.7087 | 0.5825 | Both internal and external worse |

Takeaway:

- End-to-end multimodal training improved internal performance somewhat.
- It did not improve cross-domain generalization.

### 5.7 Test-time adaptation and post-hoc correction

| Experiment | Description | Internal AUROC | External TCGA AUROC | Notes |
| --- | --- | ---: | ---: | --- |
| `exp28_exp20b_tta` | Test-time feature mean/std alignment on `exp20b` | - | 0.6353 | Best TCGA result so far |
| `exp29_exp20b_tta_ensemble` | `exp20b` + `exp28` post-hoc ensemble | - | 0.6353 | Equivalent to TTA alone |
| `exp30_exp20b_tta_tempcal` | TTA + temperature scaling + checkpoint weighting | - | 0.6280 | Calibration hurt performance |

Takeaway:

- Simple feature-stat alignment at test time gave the best external AUROC in the project so far.
- Additional probability calibration did not help.

## 6. Current best results

### 6.1 Best internal result

- Experiment: `exp01_abmil_clinpath`
- Type: post-hoc pathology + clinical ensemble
- Internal AUROC: `0.8394`

### 6.2 Best external TCGA result

- Experiment: `exp28_exp20b_tta`
- Type: TTA feature-stat alignment built on `exp20b`
- External TCGA AUROC: `0.6353`

### 6.3 Best training-time pathology-only external model

- Experiment: `exp20b_abmil_dann_w050`
- Internal AUROC: `0.7206`
- External TCGA AUROC: `0.6302`

## 7. Interpretation of current status

At this point the project has two different "best" models depending on the objective:

- If internal performance is the goal, clinical fusion is the strongest line
- If external TCGA generalization is the goal, ABMIL + DANN followed by TTA is the strongest line

This means the current failure mode is not simple underfitting.
Instead, the main issue is that the signals that help the internal cohort do not transfer well enough to TCGA.

Practical interpretation:

- Internal improvement is achievable
- External improvement is incremental and difficult
- Domain shift remains the main scientific and engineering bottleneck

## 8. Important directories for collaborators

### 8.1 Source code

- Core training and data:
  - `src/stamp/modeling/`
- Multimodal clinical code:
  - `src/stamp/modeling/clinical.py`
  - `src/stamp/modeling/models/abmil_clinical.py`
- TTA:
  - `scripts/test_time_feature_align.py`
- Post-hoc ensembling:
  - `scripts/model_ensemble.py`
- Clinical fusion:
  - `scripts/clinical_ensemble.py`
  - `scripts/clinical_transfer_ensemble.py`
  - `scripts/clinical_transfer_stack.py`
- Stain normalization:
  - `scripts/macenko_normalize_tiles.py`

### 8.2 Experiment configs

- All experiment configs are under:
  - `configs/experiments/`

Relevant late-stage configs:

- `exp20b_abmil_dann_w050_crossval.yaml`
- `exp20b_abmil_dann_w050_deploy.yaml`
- `exp21_transmil_dann_w020_crossval.yaml`
- `exp22_abmil_pseudolabel_crossval.yaml`
- `exp23_abmil_distill_crossval.yaml`
- `exp24_abmil_dann_distill_crossval.yaml`
- `exp26_abmil_clinical_joint_crossval.yaml`
- `exp27_abmil_clinical_dann_crossval.yaml`
- `exp28_exp20b_tta_stats.yaml`

### 8.3 Results

- Statistics outputs:
  - `outputs/stats/`
- Deploy predictions:
  - `outputs/deploy/`
- Cross-validation predictions and checkpoints:
  - `outputs/crossval/`

## 9. Recommended onboarding path for a new collaborator

If joining now, the recommended order is:

1. Read this document fully
2. Inspect `outputs/stats/` to understand which lines already failed and which are still promising
3. Read the following source files in order:
   - `src/stamp/modeling/config.py`
   - `src/stamp/modeling/data.py`
   - `src/stamp/modeling/crossval.py`
   - `src/stamp/modeling/train.py`
   - `src/stamp/modeling/models/__init__.py`
4. Then inspect specialized components depending on the intended direction:
   - TTA: `scripts/test_time_feature_align.py`
   - Clinical multimodal: `src/stamp/modeling/clinical.py` and `src/stamp/modeling/models/abmil_clinical.py`
   - Post-hoc model fusion: `scripts/model_ensemble.py`
5. Use `exp20b_abmil_dann_w050` and `exp28_exp20b_tta` as the main external reference points

## 10. Open questions and next suggested directions

Current evidence suggests:

- Simple post-hoc calibration is unlikely to solve the external problem
- Training-time DANN is useful but saturates early
- Clinical fusion helps internal performance but not external transfer

Promising next directions:

- Stronger pathology-model ensembles focused on external AUROC
- Better test-time adaptation that changes representations more meaningfully than mean/std alignment
- Domain generalization methods that do not depend on pseudo-label quality
- More careful source-target feature normalization strategies
- Reconsidering data splits, labels, or cohort mismatch assumptions if biologically justified

## 11. Short collaborator checklist

Before launching a new line, verify:

- Is the goal internal performance or external TCGA transfer
- Does the line require target-domain dataloaders
- Does the dataloader emit 4-item or 5-item batches
- Does the experiment produce:
  - crossval predictions
  - deploy predictions
  - stats config
  - final AUROC in `outputs/stats`

## 12. Bottom line

The project is not blocked by infrastructure anymore.
The main remaining challenge is scientific and modeling-related:

- internal-best line: `exp01_abmil_clinpath` with AUROC `0.8394`
- external-best line: `exp28_exp20b_tta` with TCGA AUROC `0.6353`

Anyone continuing from here should avoid repeating already exhausted lines unless there is a materially different hypothesis behind them.
