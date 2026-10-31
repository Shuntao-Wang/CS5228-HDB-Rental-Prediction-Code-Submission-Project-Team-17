# Competition data placement

The submitted ZIP and mirrored private code repository intentionally contain **no Kaggle training labels, test labels, auxiliary bulk files, or precomputed feature caches**.

From the official CS5228 Kaggle competition package, place these input files before running `python src/preprocessing.py`:

```text
data/raw/train.csv
data/raw/test.csv
data/raw/auxiliary/sg-hdb-block.csv
data/raw/auxiliary/sg-mrt-stations.csv
data/raw/auxiliary/sg-schools.csv
data/raw/auxiliary/sg-shopping-malls.csv
```

These are expected relative paths in `src/preprocessing.py` and are case-sensitive on some operating systems. The script builds `data/processed/train_features.csv.gz` and `data/processed/test_features.csv.gz` plus numeric model-ready CSVs.

The submission includes the competition-supplied `examples/submission_reference.csv`, used only for the `Id` column. The `Predicted` column is not an observed rental label.

Never augment test labels by looking up 2025–2026 HDB rentals from external records or using leaked predictions.
