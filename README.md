# CS5228 HDB Rental Prediction — Code Submission (Project Team 17)

This is the **minimal, self-contained source-code submission** for the CS5228 HDB monthly rent prediction Kaggle project. It intentionally excludes discarded model versions, local experiment outputs, cached features, trained checkpoints, personal environments, and any external rental-target lookups.

**Final candidate:** 90% V7 Blend5 + 10% V8 Premium predictions. Recorded Kaggle **Public** RMSE: **432.92758** (original V7 Public: 432.97923). The difference is small and not evidence that the final *Private* score is superior.

## 1. Dependencies

Python 3.10+ is recommended. Install packages in a clean environment:

```powershell
python -m pip install -r requirements.txt
```

CPU training is supported. The final configuration trains **24 individual seeded models**: V7 has five model families × three seeds (15); V8 has three families × three seeds (9). This can take substantial CPU time, especially during feature preparation. The scripts use `--threads 4` by default in the commands below.

## 2. Data (supplied by the competition, not bundled in this code submission)

Place the course-provided competition datasets under:

```text
data/
  raw/
    train.csv
    test.csv
    auxiliary/
      sg-hdb-block.csv
      sg-mrt-stations.csv
      sg-schools.csv
      sg-shopping-malls.csv
```

Any additional auxiliary files may remain in `data/raw/auxiliary/`; the final preprocessing uses the four listed files. See [data/README.md](data/README.md). **Do not fetch actual rental targets for the 2025–2026 competition test months.** The test target must remain unknown.

The included `examples/submission_reference.csv` is for **submission ID/order alignment only**. Its `Predicted` values, if present, are never treated as target labels.

## 3. Full-data prediction reproduction (run from repository root)

```powershell
# First: deterministic, feature-layer preprocessing of original inputs.
python src/preprocessing.py

# V5 (47 columns), used as the original V8 Premium base.
python scripts/v5_cache_features.py

# V7 (69 columns, including original V4-extended features and 22 ratio features).
# The intentionally value-free --taus option selects only the nondecayed historical version.
python scripts/v7_features.py --with-v5 --smooth 3 --taus --out experiments/v7/fc_v5s3nod.pkl

# V8 Premium: prepare 31 candidate extras, select only eight Premium columns downstream.
python scripts/v8_prepare_features.py

# Feature leakage/shape regression tests.
python scripts/test_v7_features.py
python scripts/test_v8_features.py

# Verify V7 feature cache and the five-member ensemble configuration.
python scripts/v7_reproduce_submission.py --check

# Five V7 model families, each averaging seeds 42, 43, and 44.
python scripts/v7_reproduce_submission.py --model cat_reg_ratio --threads 4
python scripts/v7_reproduce_submission.py --model cat_mid_ratio --threads 4
python scripts/v7_reproduce_submission.py --model lgb_reg_ratio --threads 4
python scripts/v7_reproduce_submission.py --model cat_reg_raw --threads 4
python scripts/v7_reproduce_submission.py --model cat_mid_raw --threads 4
python scripts/v7_reproduce_submission.py --blend

# V8 Premium: three model families x three seeds.
python scripts/v8_make_premium_submission.py --dry-run
python scripts/v8_make_premium_submission.py --seeds 3 --threads 4

# Create the submission CSV without any Kaggle upload.
python scripts/v7_v8_make_submission.py --check
python scripts/v7_v8_make_submission.py --write
```

Final output: **`submissions/v7_v8/blend90_10/submission.csv`**, with `Id,Predicted` and 50,000 rows. The final blending weights are fixed (V7 0.90, V8 0.10); there is no rounding.

The V7 outputs use their own provenance files and refuse to overwrite different predictions. If you are starting fresh, use the original data without pre-existing experiment caches or submissions.

## 4. Evaluation and leakage prevention

Two chronological holdouts, **not random splits**, were used:

| Split | Training approval dates | Validation approval dates |
|---|---|---|
| Long | through 2023-12 | 2024-01 to 2025-03 |
| Short | through 2024-06 | 2024-07 to 2025-03 |

All query-month historical rent features only use labels from **strictly earlier months**. Validation reference data is restricted to its training fold. V7 ratio-target forecasts depend on a level forecast estimated from the training fold only. V8 Premium transforms earlier-month V5 historical averages and does not use same-month labels.

| Candidate | Long validation RMSE | Short validation RMSE | Kaggle Public RMSE |
|---|---:|---:|---:|
| V7 Blend5 | 478.18863 | 452.65007 | 432.97923 |
| V8 Premium | 482.16171 | 454.21616 | 436.18827 |
| **Final 90:10 blend** | 478.27697 | 452.44178 | **432.92758** |

The two holdouts overlap in validation months and cannot be treated as independent samples. Relative to V7, the final blend **worsens Long by 0.08834 RMSE** and **improves Short by 0.20828**. The Public RMSE improves by **0.05165**. Private results are unknown.

For optional, computationally expensive holdout reproduction:

```powershell
# Builds historical fold predictions for each V7 family (do not add --submit).
python scripts/run_v7_experiments.py --variant cat_reg --target ratio --seeds 3 --tag best --threads 4 --cache experiments/v7/fc_v5s3nod.pkl
python scripts/run_v7_experiments.py --variant cat_mid --target ratio --seeds 3 --tag best --threads 4 --cache experiments/v7/fc_v5s3nod.pkl
python scripts/run_v7_experiments.py --variant lgb_reg --target ratio --seeds 3 --tag best --threads 4 --cache experiments/v7/fc_v5s3nod.pkl
python scripts/run_v7_experiments.py --variant cat_reg --target raw --seeds 3 --tag best --threads 4 --cache experiments/v7/fc_v5s3nod.pkl
python scripts/run_v7_experiments.py --variant cat_mid --target raw --seeds 3 --tag best --threads 4 --cache experiments/v7/fc_v5s3nod.pkl

# Builds paired V5 control / V8 Premium holdout predictions.
python scripts/run_v8_experiments.py --variant premium --blend --seeds 3 --threads 4

# Compare full V7 Blend5 to V8 and its fixed blending weights.
python scripts/v7_v8_prediction_blend.py --check
python scripts/v7_v8_prediction_blend.py --mode full
python scripts/v7_v8_full_period_audit.py
```

Use held-out `MONTHLY_RENT` only for evaluation, never to generate query-month features. The Public leaderboard score cannot be reproduced locally without Kaggle's undisclosed labels.

## 5. Source organization

- `src/preprocessing.py`: deterministic raw-input joins and feature-layer files.
- `scripts/run_v2_experiments.py`, `run_v4_experiments.py`: **shared feature-engineering utility functions**, retained as dependencies (not the selected prediction models).
- `scripts/v5_geo_features.py`, `v5_cache_features.py`, `run_v5_experiments.py`: V5 feature cache / shared model training functions (no V5 submission needed).
- `scripts/v7_features.py`, `run_v7_experiments.py`, `v7_reproduce_submission.py`: V7 features and five-family full-data predictions.
- `scripts/v8_features.py`, `v8_prepare_features.py`, `run_v8_experiments.py`, `v8_make_premium_submission.py`: V8 Premium features and predictions.
- `scripts/v7_v8_make_submission.py`: final fixed-weight prediction fusion.
- `scripts/test_v7_features.py`, `test_v8_features.py`: targeted leakage/shape checks.
- `scripts/v7_v8_prediction_blend.py`, `v7_v8_full_period_audit.py`: optional holdout comparison and temporal audit.
- `MANIFEST.sha256`: SHA-256 of every included file except itself. This helps verify that the Canvas ZIP and private GitHub mirror have identical contents.

No scripts upload to Kaggle, and no raw data, derived train/test data, cached model matrices or training artifacts are bundled.

## 6. Academic reproducibility notes

The shipped code follows the same model families, features, seeds, and blend weights as the recorded final run. Library-version and platform differences can introduce small floating-point differences; a matching score is not guaranteed to every decimal on different systems. The legitimate V7 final submission was separately trained by a second team member and scored the same **432.97923 Public RMSE** to five decimal places.

The source snapshot is mirrored in the private Team 17 GitHub code-submission repository. `MANIFEST.sha256` can be used to compare the same files after extracting the Canvas ZIP.
