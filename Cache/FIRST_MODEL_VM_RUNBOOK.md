# First Model VM Runbook

This runbook describes how to train the first CACHE PGK2 LightGBM/ECFP model and generate validation/test submission files on a VM.

## Files

The pipeline uses these files:

- `pipeline/first_model_pipeline.py` - training, scoring, and submission script.
- `pipeline/first_model_config.yaml` - full model configuration.
- `pipeline/first_model_smoke_config.yaml` - small smoke-test configuration.
- `pipeline/PGK2_train_labeled.parquet` - deduped/labeled DEL training subset.
- `Val-Test-set/PGK2_Validation_split.csv` - validation candidates.
- `Val-Test-set/PGK2_Test_split.csv` - test candidates.

## Environment

Use the `openadmet-models` conda environment.

```bash
conda activate openadmet-models
```

Check required packages:

```bash
python - <<'PY'
mods = ['pandas', 'pyarrow', 'rdkit', 'numpy', 'sklearn', 'lightgbm', 'yaml', 'joblib']
for m in mods:
    try:
        mod = __import__(m)
        print(m, 'OK', getattr(mod, '__version__', ''))
    except Exception as e:
        print(m, 'MISSING', type(e).__name__, e)
PY
```

All packages should print `OK`.

## Before Running

Edit `pipeline/first_model_config.yaml` and replace:

```yaml
team_name: Team_YOURTEAMNAME
```

with your real challenge team filename prefix.

You can also adjust:

```yaml
run:
  n_jobs: -1
  top_pool_size: 5000
  final_pick_count: 50
```

Use `n_jobs: -1` to use all CPU cores. Use a smaller value if the VM is shared or memory-limited.

## Smoke Test

Run the smoke test first. It uses only a small sample and should finish quickly.

```bash
python pipeline/first_model_pipeline.py \
  --config pipeline/first_model_smoke_config.yaml
```

Expected output directory:

```text
pipeline/first_model_smoke_outputs/
```

Important smoke-test files:

```text
pipeline/first_model_smoke_outputs/holdout_metrics.json
pipeline/first_model_smoke_outputs/validation_selected_top50.csv
pipeline/first_model_smoke_outputs/test_selected_top50.csv
pipeline/first_model_smoke_outputs/Team_YOURTEAMNAME_smoke_lgbm_ecfp_ensemble_v1_validation.txt
pipeline/first_model_smoke_outputs/Team_YOURTEAMNAME_smoke_lgbm_ecfp_ensemble_v1_test.csv
```

Check output format:

```bash
wc -l pipeline/first_model_smoke_outputs/*_validation.txt

python - <<'PY'
import pandas as pd
p = 'pipeline/first_model_smoke_outputs/Team_YOURTEAMNAME_smoke_lgbm_ecfp_ensemble_v1_test.csv'
df = pd.read_csv(p)
print(df.shape)
print(df.columns.tolist())
print('Sel_50 sum:', int(df.Sel_50.sum()))
PY
```

The validation file should have `50` lines. The test file should have columns:

```text
CatalogID, Sel_50, Score
```

and `Sel_50 sum` should be `50`.

Do not submit smoke-test outputs.

## Full Run

Run the full model:

```bash
python pipeline/first_model_pipeline.py \
  --config pipeline/first_model_config.yaml
```

Expected output directory:

```text
pipeline/first_model_outputs/
```

Main output files:

```text
pipeline/first_model_outputs/holdout_metrics.json
pipeline/first_model_outputs/run_summary.json
pipeline/first_model_outputs/validation_scores.parquet
pipeline/first_model_outputs/test_scores.parquet
pipeline/first_model_outputs/validation_selected_top50.csv
pipeline/first_model_outputs/test_selected_top50.csv
pipeline/first_model_outputs/Team_YOURTEAMNAME_lgbm_ecfp_ensemble_v1_validation.txt
pipeline/first_model_outputs/Team_YOURTEAMNAME_lgbm_ecfp_ensemble_v1_test.csv
```

The two challenge-format files are:

```text
Team_YOURTEAMNAME_lgbm_ecfp_ensemble_v1_validation.txt
Team_YOURTEAMNAME_lgbm_ecfp_ensemble_v1_test.csv
```

## What The Model Does

The script trains four LightGBM models:

| Model | Target |
|---|---|
| `broad` | Current broad binary PGK2 hit label, `LABEL == 1` |
| `strict` | Higher-confidence hits with `count_PGK2 >= 10`, `count_NTC == 0`, `historic_hits < 5` |
| `orthosteric` | PGK2 hits blocked by inhibitor, approximated by `count_PGK2_with_inhibitor == 0` |
| `zscore` | Regression against `zscore_PGK2` |

It scores validation/test candidates, converts each model score to a percentile rank, combines the ranks using the YAML ensemble weights, then selects a diverse top 50 using Murcko scaffold and ECFP Tanimoto constraints.

## Submission Caution

Before submitting:

1. Open `validation_selected_top50.csv` and `test_selected_top50.csv`.
2. Confirm there are 50 compounds.
3. Check that the selected molecules are not all the same chemotype.
4. Confirm the team name in the filenames is correct.
5. Submit the validation file first if you want exploratory feedback.
6. Use test submissions carefully because the challenge allows only two scored test submissions in the blind phase.

## Useful Config Knobs

Increase diversity:

```yaml
selection:
  max_per_murcko_scaffold: 1
  min_tanimoto_distance: 0.35
```

Allow more close analogs:

```yaml
selection:
  max_per_murcko_scaffold: 2
  min_tanimoto_distance: 0.15
```

Change model weights:

```yaml
ensemble:
  broad: 0.40
  strict: 0.25
  orthosteric: 0.20
  zscore: 0.15
```

For a first validation submission, keep the default diversity-aware setting. For later validation experiments, create copies of the YAML file with different `submission_id`, `ensemble`, and `selection` settings.
