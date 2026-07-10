# `first_model_pipeline.py` Explained

This document explains what `pipeline/first_model_pipeline.py` does and how it turns the processed CACHE PGK2 data into validation/test submission files.

## Purpose

`first_model_pipeline.py` is the first executable predictive-model pipeline for the CACHE PGK2 challenge.

It does five main things:

1. Loads the processed DEL training data and ASMS validation/test candidate files.
2. Converts SMILES into ECFP fingerprints.
3. Trains several LightGBM models using different PGK2-related target definitions.
4. Scores validation and test candidates.
5. Selects 50 chemically diverse compounds and writes challenge-format submission files.

The script is controlled by a YAML file, usually:

```bash
pipeline/first_model_config.yaml
```

Run command:

```bash
python pipeline/first_model_pipeline.py \
  --config pipeline/first_model_config.yaml
```

## Input Files

The pipeline expects:

```text
pipeline/PGK2_train_labeled.parquet
Val-Test-set/PGK2_Validation_split.csv
Val-Test-set/PGK2_Test_split.csv
```

The training Parquet file contains deduped DEL molecules with labels and selection statistics:

```text
SMILES
compound
count_PGK2
count_PGK2_with_inhibitor
count_NTC
zscore_PGK2
zscore_PGK2_with_inhibitor
zscore_NTC
historic_hits
LABEL
```

The validation/test CSV files contain:

```text
CatalogID
SMILES
```

## High-Level Data Flow

The script follows this sequence:

```text
Load YAML config
  |
Load training/validation/test files
  |
Optionally sample rows for smoke testing
  |
Compute Murcko scaffolds
  |
Create scaffold-aware holdout split
  |
For each target:
    Build target labels
    Compute ECFP features
    Train LightGBM model
    Evaluate on scaffold holdout
    Save model
  |
Score validation candidates
Score test candidates
  |
Combine model scores into ensemble score
  |
Select diverse top 50 compounds
  |
Write selected tables and submission files
```

## Configuration

The YAML config controls paths, model settings, targets, ensemble weights, and selection rules.

Important sections:

```yaml
paths:
  train_labeled: pipeline/PGK2_train_labeled.parquet
  validation_csv: Val-Test-set/PGK2_Validation_split.csv
  test_csv: Val-Test-set/PGK2_Test_split.csv
  output_dir: pipeline/first_model_outputs
```

These tell the script where to read data and where to write outputs.

```yaml
run:
  seed: 42
  n_jobs: -1
  top_pool_size: 5000
  final_pick_count: 50
  team_name: Team_YOURTEAMNAME
  submission_id: lgbm_ecfp_ensemble_v1
```

These control reproducibility, CPU usage, final selection size, and output filenames.

```yaml
features:
  fp_size: 2048
  radii: [2, 3]
```

This means the model uses:

- Morgan radius 2 fingerprints, equivalent to ECFP4.
- Morgan radius 3 fingerprints, equivalent to ECFP6.
- 2048 bits per fingerprint radius.

The final feature matrix has `4096` columns because ECFP4 and ECFP6 are concatenated.

## Step 1: Load Data

The script reads:

```python
train = pd.read_parquet(cfg["paths"]["train_labeled"])
validation = pd.read_csv(cfg["paths"]["validation_csv"])
test = pd.read_csv(cfg["paths"]["test_csv"])
```

The training set is used to fit models. The validation/test files are only scored and ranked.

## Step 2: Optional Debug Sampling

The config can specify row limits:

```yaml
debug_max_train_rows:
debug_max_validation_rows:
debug_max_test_rows:
```

In the full config these are empty, so the full data are used.

In the smoke-test config they are set to small values. This lets you test the pipeline quickly before running the full model.

The training debug sampler preserves the approximate class balance of `LABEL`.

## Step 3: Compute Murcko Scaffolds

For each molecule, the script computes a Bemis-Murcko scaffold using RDKit:

```python
MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
```

The scaffold is stored as:

```text
murcko_scaffold
```

Scaffolds are used for two things:

1. Creating a scaffold-aware internal holdout split.
2. Selecting a chemically diverse final top 50.

This matters because random splits are misleading for DEL data. Similar analogs can leak across random train/test splits and inflate performance.

## Step 4: Scaffold-Aware Holdout Split

The script creates an internal holdout set by scaffold, not by random molecule:

```python
make_scaffold_split(...)
```

The config controls the holdout fraction:

```yaml
split:
  scaffold_holdout_fraction: 0.20
```

This means roughly 20% of training rows are held out by scaffold identity.

The goal is not to perfectly mimic the real ASMS distribution shift, but it is much better than a random split.

## Step 5: Build ECFP Features

For every molecule, the script computes Morgan fingerprints using RDKit:

```python
rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=fp_size)
```

With the default config:

```yaml
radii: [2, 3]
fp_size: 2048
```

the script creates:

```text
ECFP4 2048-bit vector
ECFP6 2048-bit vector
```

and concatenates them into one sparse matrix:

```text
4096 fingerprint features per molecule
```

The matrix is stored as SciPy CSR sparse format. Values are `float32` because this LightGBM build expects sparse values to be floating point.

## Step 6: Define Four Training Targets

The script trains four different models. Each model captures a different interpretation of PGK2 activity.

### 1. Broad Classifier

Config:

```yaml
broad:
  kind: classifier
  include: broad_all
  positive:
    label_column: LABEL
    label_value: 1
```

This uses the existing broad hit label:

```text
LABEL == 1
```

This is the main baseline model.

It includes:

- all current hits as positives
- sampled non-hits as negatives

### 2. Strict Classifier

Config:

```yaml
strict:
  kind: classifier
  include: strict_pos_and_broad_neg
  positive:
    min_count_pgk2: 10
    max_count_ntc: 0
    max_historic_hits: 5
```

This model learns from higher-confidence DEL hits.

A strict positive must satisfy:

```text
count_PGK2 >= 10
count_NTC <= 0
historic_hits < 5
```

It uses strict positives plus broad negatives.

This model is intentionally narrower than the broad model. It may rank stronger enrichment patterns higher.

### 3. Orthosteric Classifier

Config:

```yaml
orthosteric:
  kind: classifier
  include: orthosteric_pos_and_broad_neg
  positive:
    min_count_pgk2: 3
    max_count_pgk2_with_inhibitor: 0
    max_count_ntc: 0
    max_historic_hits: 5
```

This approximates ATP-site/orthosteric binders.

A positive must satisfy:

```text
count_PGK2 >= 3
count_PGK2_with_inhibitor <= 0
count_NTC <= 0
historic_hits < 5
```

The logic is that if the PGK2 signal disappears in the presence of a known ATP-site inhibitor, the compound may compete at or near the ATP site.

This is not a perfect mechanistic label, but it adds a useful orthosteric-biased signal to the ensemble.

### 4. Z-Score Regressor

Config:

```yaml
zscore:
  kind: regressor
  include: broad_all
  target_column: zscore_PGK2
```

This model predicts the continuous PGK2 enrichment score:

```text
zscore_PGK2
```

It gives the ensemble a continuous enrichment signal instead of only binary thresholds.

## Step 7: Train LightGBM Models

Classifier models use:

```python
LGBMClassifier(
    objective="binary",
    class_weight="balanced",
)
```

The regressor uses:

```python
LGBMRegressor(
    objective="regression",
)
```

Hyperparameters come from YAML:

```yaml
models:
  classifier:
    n_estimators: 900
    learning_rate: 0.035
    num_leaves: 63
    ...
```

The trained models are saved as:

```text
model_broad.joblib
model_strict.joblib
model_orthosteric.joblib
model_zscore.joblib
```

## Step 8: Internal Holdout Evaluation

Each model is evaluated on the scaffold holdout set.

For classifiers, the script reports:

```text
ROC-AUC
PR-AUC
top50_hits
top50_distinct_scaffolds
```

For the z-score regressor, it reports:

```text
RMSE
broad ROC-AUC
broad PR-AUC
top50_hits
top50_distinct_scaffolds
```

These metrics are saved to:

```text
holdout_metrics.json
```

Important caution: these metrics are useful diagnostics, but they are not the real competition score. The ASMS validation/test sets are much more out-of-distribution than the internal holdout.

## Step 9: Score Validation And Test Candidates

After training, the script computes fingerprints for:

```text
Val-Test-set/PGK2_Validation_split.csv
Val-Test-set/PGK2_Test_split.csv
```

Each model scores each candidate.

The output score columns are:

```text
score_broad
score_strict
score_orthosteric
score_zscore
```

Full scored tables are saved as:

```text
validation_scores.parquet
test_scores.parquet
```

## Step 10: Ensemble Scoring

Raw model scores are on different scales. For example:

- classifier outputs are probabilities
- z-score regressor outputs are predicted z-scores

So the script converts each model's raw score into a percentile rank:

```text
lowest score  -> near 0
highest score -> near 1
```

Then it combines percentile ranks using YAML weights:

```yaml
ensemble:
  broad: 0.40
  strict: 0.25
  orthosteric: 0.20
  zscore: 0.15
```

The final score is:

```text
ensemble_score =
  0.40 * rank(score_broad)
+ 0.25 * rank(score_strict)
+ 0.20 * rank(score_orthosteric)
+ 0.15 * rank(score_zscore)
```

This rank-based ensemble is robust because it does not assume the four models are calibrated on the same probability scale.

## Step 11: Diversity-Aware Top-50 Selection

The challenge rewards chemically distinct hit series, not just raw hit count. Therefore the script does not simply take the top 50 ensemble scores.

Instead, it:

1. Sorts all candidates by `ensemble_score`.
2. Keeps the top pool, default `5000`.
3. Iterates from highest to lower score.
4. Selects compounds while enforcing diversity constraints.

Default selection config:

```yaml
selection:
  max_per_murcko_scaffold: 1
  min_tanimoto_distance: 0.25
```

This means:

- Prefer no more than one selected compound per Murcko scaffold.
- Prefer selected compounds with ECFP Tanimoto similarity no greater than `0.75`.

The script uses a fallback system:

1. First pass: enforce both scaffold and Tanimoto diversity.
2. Second pass: relax Tanimoto diversity but keep scaffold diversity.
3. Third pass: relax both if needed to reach 50 compounds.

This ensures the output always contains 50 selected compounds.

## Step 12: Write Output Files

The script writes inspection tables:

```text
validation_selected_top50.csv
test_selected_top50.csv
```

These contain:

```text
selection_rank
CatalogID
SMILES
ensemble_score
murcko_scaffold
score_broad
score_strict
score_orthosteric
score_zscore
```

These files are for human review before submission.

## Step 13: Write Challenge Submission Files

The validation submission file is a text file with 50 CatalogIDs:

```text
Team_YOURTEAMNAME_lgbm_ecfp_ensemble_v1_validation.txt
```

The test submission file is a CSV:

```text
Team_YOURTEAMNAME_lgbm_ecfp_ensemble_v1_test.csv
```

with columns:

```text
CatalogID, Sel_50, Score
```

`Sel_50` is:

```text
1 for the selected 50 molecules
0 for all other test molecules
```

`Score` is the final `ensemble_score`.

## Smoke-Test Config

The smoke-test config:

```text
pipeline/first_model_smoke_config.yaml
```

uses:

```yaml
debug_max_train_rows: 5000
debug_max_validation_rows: 2000
debug_max_test_rows: 2000
```

and smaller fingerprints/models.

It is only for checking that the environment, file paths, and output generation work.

Do not submit smoke-test outputs.

## Important Limitations

This is a strong first baseline, but it is not the final ideal workflow.

Known limitations:

1. It trains only on the existing 82,926-row labeled subset, not the full deduped DEL pool.
2. The internal holdout is scaffold-aware, but still DEL-derived and not truly ASMS-like.
3. It uses only ECFP fingerprints, not Chemprop or MMELON embeddings.
4. It does not yet use docking, shape, pharmacophore, or protein-structure information.
5. It treats sampled DEL non-hits as negatives, although some may be false negatives.
6. The orthosteric label is an approximation based on inhibitor competition counts.

These limitations are acceptable for a first submission-generating model, but later models should add Chemprop/MMELON and use validation feedback to improve selection.

## Practical Review Checklist

Before submitting files generated by the full run:

1. Confirm `team_name` in the YAML is correct.
2. Confirm the validation file has exactly 50 lines.
3. Confirm the test file has exactly 50 rows with `Sel_50 == 1`.
4. Inspect `validation_selected_top50.csv`.
5. Inspect `test_selected_top50.csv`.
6. Check that selected molecules are chemically diverse.
7. Submit validation first if you want exploratory feedback.
8. Use blind test submissions carefully because only two are allowed.

## Summary

`first_model_pipeline.py` is a complete first-pass CACHE submission pipeline. It trains multiple ECFP/LightGBM models against complementary PGK2 target definitions, ensembles their ranks, and selects a chemically diverse top 50 for validation and test submission.

Its main design choice is pragmatic: optimize for transferable ranking and distinct chemical series, not just internal AUC.
