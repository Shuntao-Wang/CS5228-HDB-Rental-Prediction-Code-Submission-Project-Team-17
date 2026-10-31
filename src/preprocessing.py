"""CS5228 HDB rental — reproducible V1 basic + geospatial feature generation.

Run from any directory:
    python src/preprocessing.py

Writes two matched versions to data/processed/:
  - train_features.csv.gz / test_features.csv.gz: deterministic features, category
    strings preserved for fold-safe modeling and CatBoost-style models;
  - train_processed.csv.gz / test_processed.csv.gz: ready-to-use numeric CSVs.

Numeric encoding / imputation is fitted on FULL competition training data.
Therefore these numeric files are appropriate for final Kaggle train->test runs,
NOT for a leakage-free offline time holdout. For validation, use *_features.csv.gz
and fit any category vocabulary/imputation on the fold's training portion only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"

EARTH_RADIUS_KM = 6371.0088
CATEGORICAL = ["TOWN", "FLAT_TYPE", "FLAT_MODEL", "REGION", "PLANNING_AREA"]
NUMERIC = [
    "APPROVAL_YEAR", "APPROVAL_MONTH", "MONTH_INDEX", "MONTH_SIN", "MONTH_COS",
    "FLOOR_AREA_SQM", "LEASE_COMMENCE_DATE", "LEASE_AGE", "LEASE_AGE_INVALID",
    "LATITUDE", "LONGITUDE", "MAX_FLOOR", "YEAR_COMPLETED", "BUILDING_AGE",
    "MAX_FLOOR_MISSING", "YEAR_COMPLETED_MISSING",
    "MRT_NEAREST_KM", "MRT_WITHIN_1KM",
    "PRIMARY_SCHOOL_NEAREST_KM", "PRIMARY_SCHOOL_WITHIN_1KM",
    "MALL_NEAREST_KM", "MALL_WITHIN_2KM",
]
# Extra columns retained for fold-aware feature engineering and auditing only.
EXTRA = ["RENT_APPROVAL_DATE", "STREET", "BLOCK_KEY"]


def normalize(s: pd.Series) -> pd.Series:
    """Normalize case and spaces, without treating numbers as numeric indices."""
    return s.astype("string").str.strip().str.casefold().str.replace(r"\s+", " ", regex=True)


def flat_type_normalized(s: pd.Series) -> pd.Series:
    s = normalize(s)
    return s.str.replace(r"^(\d+)\s*[- ]\s*room$", r"\1-room", regex=True)


def add_join_keys(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in ["TOWN", "BLOCK", "STREET"]:
        df[c] = normalize(df[c])
    return df


def haversine_km(coords_a: np.ndarray, coords_b: np.ndarray) -> np.ndarray:
    """Distances between N HDB coords and M POI coords, NxM matrix in km."""
    lat_a = np.deg2rad(coords_a[:, 0])[:, None]
    lon_a = np.deg2rad(coords_a[:, 1])[:, None]
    lat_b = np.deg2rad(coords_b[:, 0])[None, :]
    lon_b = np.deg2rad(coords_b[:, 1])[None, :]
    delta_lat = lat_b - lat_a
    delta_lon = lon_b - lon_a
    a = (np.sin(delta_lat / 2) ** 2 +
         np.cos(lat_a) * np.cos(lat_b) * np.sin(delta_lon / 2) ** 2)
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def poi_location_table(path: Path, only_open: bool = False,
                       only_primary: bool = False) -> pd.DataFrame:
    poi = pd.read_csv(path)
    if only_open:
        poi = poi.loc[normalize(poi["STATUS"]) == "open"]
    if only_primary:
        poi = poi.loc[normalize(poi["MAINLEVEL_CODE"]) == "primary"]
    poi = poi[["LATITUDE", "LONGITUDE"]].apply(pd.to_numeric, errors="coerce")
    return poi.dropna().drop_duplicates().reset_index(drop=True)


def add_geospatial_features(data: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Calculate on distinct HDB coordinates to avoid repeating work."""
    coords = data[["LATITUDE", "LONGITUDE"]].drop_duplicates().reset_index(drop=True)
    if coords.isna().any().any():
        raise ValueError("HDB block latitude/longitude is missing; inspect join.")
    location_arrays = {
        "MRT": poi_location_table(RAW / "auxiliary" / "sg-mrt-stations.csv", only_open=True),
        "PRIMARY_SCHOOL": poi_location_table(RAW / "auxiliary" / "sg-schools.csv", only_primary=True),
        "MALL": poi_location_table(RAW / "auxiliary" / "sg-shopping-malls.csv"),
    }
    thresholds = {"MRT": 1.0, "PRIMARY_SCHOOL": 1.0, "MALL": 2.0}
    stats: dict[str, int] = {k + "_unique_poi_coordinates": len(v) for k, v in location_arrays.items()}
    for label, poi in location_arrays.items():
        if len(poi) == 0:
            raise ValueError(f"No valid points of interest for {label}")
        distances = haversine_km(coords.to_numpy(dtype=float), poi.to_numpy(dtype=float))
        radius_km = thresholds[label]
        coords[f"{label}_NEAREST_KM"] = distances.min(axis=1).round(5)
        radius = str(int(radius_km))
        coords[f"{label}_WITHIN_{radius}KM"] = (distances <= radius_km).sum(axis=1).astype("int16")
    enriched = data.merge(coords, how="left", on=["LATITUDE", "LONGITUDE"], validate="many_to_one", sort=False)
    if len(enriched) != len(data):
        raise AssertionError("Geospatial enrichment unexpectedly changed row count")
    return enriched, stats


def build_features(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    expected_test = set(train.columns) - {"MONTHLY_RENT"}
    if set(test.columns) != expected_test:
        raise ValueError("Train/Test column mismatch; unexpected dataset schema")
    if train["MONTHLY_RENT"].isna().any():
        raise ValueError("Training labels contain NaN")

    n_train, n_test = len(train), len(test)
    y = pd.to_numeric(train["MONTHLY_RENT"], errors="raise").reset_index(drop=True)
    data = pd.concat([train.drop(columns="MONTHLY_RENT"), test], ignore_index=True)
    data = add_join_keys(data)
    data["FLAT_TYPE"] = flat_type_normalized(data["FLAT_TYPE"])
    data["FLAT_MODEL"] = normalize(data["FLAT_MODEL"])
    data["BLOCK_KEY"] = data["TOWN"] + "|" + data["BLOCK"] + "|" + data["STREET"]

    approval = pd.to_datetime(data["RENT_APPROVAL_DATE"], format="%Y-%m", errors="raise")
    data["APPROVAL_YEAR"] = approval.dt.year.astype("int16")
    data["APPROVAL_MONTH"] = approval.dt.month.astype("int8")
    data["MONTH_INDEX"] = ((data["APPROVAL_YEAR"] - 2021) * 12 + data["APPROVAL_MONTH"] - 1).astype("int16")
    data["MONTH_SIN"] = np.sin(2 * np.pi * data["APPROVAL_MONTH"] / 12)
    data["MONTH_COS"] = np.cos(2 * np.pi * data["APPROVAL_MONTH"] / 12)
    data["FLOOR_AREA_SQM"] = pd.to_numeric(data["FLOOR_AREA_SQM"], errors="raise")
    data["LEASE_COMMENCE_DATE"] = pd.to_numeric(data["LEASE_COMMENCE_DATE"], errors="raise")
    data["LEASE_AGE"] = data["APPROVAL_YEAR"] - data["LEASE_COMMENCE_DATE"]
    data["LEASE_AGE_INVALID"] = (data["LEASE_AGE"] < 0).astype("int8")
    data.loc[data["LEASE_AGE_INVALID"] == 1, "LEASE_AGE"] = np.nan

    hdb = add_join_keys(pd.read_csv(RAW / "auxiliary" / "sg-hdb-block.csv", dtype={"POSTAL_CODE": "string"}))
    join_cols = ["TOWN", "BLOCK", "STREET"]
    if hdb.duplicated(join_cols).any():
        raise ValueError("HDB block auxiliary join keys are not unique")
    hdb_cols = ["LATITUDE", "LONGITUDE", "MAX_FLOOR", "YEAR_COMPLETED", "REGION", "PLANNING_AREA"]
    hdb = hdb[join_cols + hdb_cols]
    data = data.merge(hdb, how="left", on=join_cols, validate="many_to_one", indicator=True, sort=False)
    matches = data["_merge"].eq("both")
    match_rate_train = matches.iloc[:n_train].mean()
    match_rate_test = matches.iloc[n_train:].mean()
    if not matches.all():
        raise ValueError(f"HDB block match incomplete: train={match_rate_train:.5f}, test={match_rate_test:.5f}")
    data.drop(columns="_merge", inplace=True)

    for col in ["LATITUDE", "LONGITUDE", "MAX_FLOOR", "YEAR_COMPLETED"]:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    for col in ["MAX_FLOOR", "YEAR_COMPLETED"]:
        data[col] = data[col].replace(-1, np.nan)
        data[col + "_MISSING"] = data[col].isna().astype("int8")
    data["BUILDING_AGE"] = data["APPROVAL_YEAR"] - data["YEAR_COMPLETED"]
    for col in ["REGION", "PLANNING_AREA"]:
        data[col] = normalize(data[col])
    # Raw constants are intentionally excluded from the emitted features.
    data, poi_stats = add_geospatial_features(data)

    columns = NUMERIC + CATEGORICAL + EXTRA
    features = data.loc[:, columns].copy()
    train_features = features.iloc[:n_train].reset_index(drop=True).copy()
    test_features = features.iloc[n_train:].reset_index(drop=True).copy()
    train_features["MONTHLY_RENT"] = y
    assert len(train_features) == n_train and len(test_features) == n_test
    assert train_features.drop(columns="MONTHLY_RENT").columns.equals(test_features.columns)

    report = {
        "train_rows": n_train, "test_rows": n_test,
        "hdb_block_match_rate_train": float(match_rate_train),
        "hdb_block_match_rate_test": float(match_rate_test),
        "sentinel_missing_train": {c: int(train_features[c + "_MISSING"].sum()) for c in ["MAX_FLOOR", "YEAR_COMPLETED"]},
        "sentinel_missing_test": {c: int(test_features[c + "_MISSING"].sum()) for c in ["MAX_FLOOR", "YEAR_COMPLETED"]},
        "invalid_lease_age": int(features["LEASE_AGE_INVALID"].sum()),
        "negative_building_age": int((features["BUILDING_AGE"] < 0).sum()),
        "n_deterministic_features": len(columns),
        "poi": poi_stats,
        "train_approval_date_min": str(train_features["RENT_APPROVAL_DATE"].min()),
        "train_approval_date_max": str(train_features["RENT_APPROVAL_DATE"].max()),
        "test_approval_date_min": str(test_features["RENT_APPROVAL_DATE"].min()),
        "test_approval_date_max": str(test_features["RENT_APPROVAL_DATE"].max()),
    }
    return train_features, test_features, report


def numeric_ready_features(train_f: pd.DataFrame, test_f: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Fit category vocabulary and medians on FULL train for final Kaggle use.

    Not suitable for strictly fold-safe time-validation; see module docstring.
    """
    for col in NUMERIC:
        if not (pd.api.types.is_numeric_dtype(train_f[col]) and pd.api.types.is_numeric_dtype(test_f[col])):
            raise TypeError(f"Expected numeric column {col}")
    tr_n = train_f[NUMERIC].copy()
    te_n = test_f[NUMERIC].copy()
    medians = tr_n.median(numeric_only=True)
    if medians.isna().any():
        raise ValueError("Entirely missing numeric training column: " + str(medians[medians.isna()].index.tolist()))
    tr_n = tr_n.fillna(medians)
    te_n = te_n.fillna(medians)

    # Fit category one-hot vocabulary on training data only; unseen test categories all-zero.
    tr_c = pd.get_dummies(train_f[CATEGORICAL].fillna("unknown"), prefix=CATEGORICAL, dtype="int8")
    te_c = pd.get_dummies(test_f[CATEGORICAL].fillna("unknown"), prefix=CATEGORICAL, dtype="int8")
    unseen_test_columns = sorted(set(te_c.columns) - set(tr_c.columns))
    te_c = te_c.reindex(columns=tr_c.columns, fill_value=0)

    train_out = pd.concat([tr_n, tr_c], axis=1)
    test_out = pd.concat([te_n, te_c], axis=1)
    train_out["MONTHLY_RENT"] = train_f["MONTHLY_RENT"].to_numpy()
    if not np.isfinite(train_out.drop(columns="MONTHLY_RENT").to_numpy(dtype=float)).all():
        raise AssertionError("Non-finite values in numeric training features")
    if not np.isfinite(test_out.to_numpy(dtype=float)).all():
        raise AssertionError("Non-finite values in numeric test features")
    if not train_out.drop(columns="MONTHLY_RENT").columns.equals(test_out.columns):
        raise AssertionError("Processed train/test columns differ")
    return train_out, test_out, {
        "numeric_feature_count": len(NUMERIC),
        "onehot_feature_count": len(tr_c.columns),
        "model_feature_count": len(test_out.columns),
        "test_only_categories_ignored": unseen_test_columns,
        "train_median_fill_values": {k: float(v) for k, v in medians.items() if tr_n[k].isna().sum() > 0 or train_f[k].isna().sum() > 0},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate V1 basic + geo CS5228 HDB features")
    parser.add_argument("--output-dir", type=Path, default=OUT)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(RAW / "train.csv", dtype={"BLOCK": "string"})
    test = pd.read_csv(RAW / "test.csv", dtype={"BLOCK": "string"})
    tr_f, te_f, report = build_features(train, test)
    tr_n, te_n, numeric_report = numeric_ready_features(tr_f, te_f)
    report.update(numeric_report)
    outputs = {
        "train_features.csv.gz": tr_f,
        "test_features.csv.gz": te_f,
        "train_processed.csv.gz": tr_n,
        "test_processed.csv.gz": te_n,
    }
    for name, df in outputs.items():
        path = out / name
        df.to_csv(path, index=False, compression="gzip", float_format="%.6f")
        print(f"Wrote {path.relative_to(ROOT) if path.is_relative_to(ROOT) else path}: {df.shape}")
    metadata_path = out / "preprocessing_metadata.json"
    metadata_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {metadata_path}")
    print(f"Model-ready feature count: {numeric_report['model_feature_count']}")


if __name__ == "__main__":
    main()
