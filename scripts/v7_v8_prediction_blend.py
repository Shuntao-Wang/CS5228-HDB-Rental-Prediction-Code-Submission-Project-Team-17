"""Check and compare V7 vs V8 Premium predictions on chronological holdouts.

Commands (Windows PowerShell, from repository root):
  python scripts/v7_v8_prediction_blend.py --check
  python scripts/v7_v8_prediction_blend.py --mode quick
  python scripts/v7_v8_prediction_blend.py --mode full

Quick screening: two ORIGINAL V7 cat_mid single-seed predictors (ratio/raw),
paired against the precomputed V8 Premium blend3_s3. This is not V7 Blend5.

Full comparison: five ORIGINAL V7 3-seed models, equal averaged (the V7 Blend5
configuration), paired against the same precomputed V8 Premium blend3_s3.

Uses saved legitimate Long/Short validation labels and predictions ONLY.
No Kaggle test labels, pseudo-labels, uploaded submissions, or suspect files.
All inputs are checked for row, date, label, and finite-value agreement.
The evaluated mixing weights (0, .10, .20, .30 V8) are fixed diagnostic values,
not automatically tuned or selected. No retraining or Kaggle upload.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from run_v2_experiments import ROOT

V8_ROOT = ROOT / "experiments/v8/runs/premium_blend3_s3"
QUICK_ROOT = ROOT / "experiments/v7_premium/runs"
V7_ROOT = ROOT / "experiments/v7/runs"
OUT = ROOT / "experiments/v7_v8_blend"
WEIGHTS = (0.0, 0.1, 0.2, 0.3)
FULL_MODELS = (
    ("cat_reg", "ratio"),
    ("cat_mid", "ratio"),
    ("lgb_reg", "ratio"),
    ("cat_reg", "raw"),
    ("cat_mid", "raw"),
)
EXPECTED_ROWS = {"long": 44163, "short": 25984}


def v8_path(split):
    return V8_ROOT / f"valid_{split}.csv"


def quick_path(target, split):
    return QUICK_ROOT / f"cat_mid_{target}_s1" / f"valid_{split}.csv"


def full_path(model, target, split):
    return V7_ROOT / f"{model}_{target}_s3_best" / f"valid_{split}.csv"


def files_for(mode):
    files = []
    for split in ("long", "short"):
        files.append(v8_path(split))
        if mode == "quick":
            files.extend(quick_path(target, split) for target in ("ratio", "raw"))
        elif mode == "full":
            files.extend(full_path(model, target, split) for model, target in FULL_MODELS)
        else:
            raise ValueError(mode)
    return files


def check(mode):
    print(f"\n[{mode.upper()}] Required stored validation predictions:")
    missing = []
    for path in files_for(mode):
        exists = path.is_file()
        print(f"  {'OK     ' if exists else 'MISSING'} {path.relative_to(ROOT)}")
        if not exists:
            missing.append(path)
    if not missing:
        print(f"{mode.upper()} DATA READY: all required prediction CSV files found.")
    else:
        print(f"{mode.upper()} DATA INCOMPLETE: {len(missing)} prediction file(s) missing.")
        if mode == "full" and all(str(p).startswith(str(V7_ROOT)) for p in missing):
            print("To generate missing V7 3-seed holdout predictions (do not add --submit):")
            for model, target in FULL_MODELS:
                if any(not full_path(model, target, sp).is_file() for sp in ("long", "short")):
                    print(
                        "  python scripts/run_v7_experiments.py"
                        f" --variant {model} --target {target} --seeds 3 --tag best"
                        " --threads 4 --cache experiments/v7/fc_v5s3nod.pkl"
                    )
        if any(str(p).startswith(str(V8_ROOT)) for p in missing):
            print("The V8 Premium validation predictions may be missing after a folder cleanup.")
            print("Check whether experiments/v8/runs/premium_blend3_s3/valid_*.csv are still on this PC.")
    return not missing


def read(path, column, split):
    d = pd.read_csv(path)
    if not {"month", "actual", column}.issubset(d.columns):
        raise ValueError(f"Missing required columns in {path}")
    if len(d) != EXPECTED_ROWS[split]:
        raise ValueError(f"Unexpected row count in {path}: {len(d)}")
    for col in ("actual", column):
        if not np.isfinite(pd.to_numeric(d[col], errors="coerce").to_numpy(dtype=float)).all():
            raise ValueError(f"Nonfinite {col} in {path}")
    return d


def checked(d, reference, path):
    if not np.array_equal(d["month"].astype(str).to_numpy(), reference["month"].astype(str).to_numpy()):
        raise ValueError(f"Validation month/row order mismatch: {path}")
    if not np.allclose(d["actual"].to_numpy(float), reference["actual"].to_numpy(float),
                       atol=1e-4, rtol=0):
        raise ValueError(f"Validation labels/row order mismatch: {path}")


def rmse(y, p):
    return float(np.sqrt(np.mean((np.asarray(y, dtype=float) - p) ** 2)))


def evaluate(mode):
    if not check(mode):
        print(f"\nCannot evaluate {mode} until required files are present.")
        return
    records = []
    for split in ("long", "short"):
        v8file = v8_path(split)
        v8 = read(v8file, "pred_v8", split)
        ys = v8["actual"].to_numpy(float)
        p8 = v8["pred_v8"].to_numpy(float)
        v7_preds = []
        if mode == "quick":
            inputs = [(quick_path(target, split), "pred_v7") for target in ("ratio", "raw")]
        else:
            inputs = [(full_path(model, target, split), "pred")
                      for model, target in FULL_MODELS]
        for path, col in inputs:
            d = read(path, col, split)
            checked(d, v8, path)
            v7_preds.append(d[col].to_numpy(float))
        p7 = np.mean(np.stack(v7_preds), axis=0)
        base = rmse(ys, p7)
        print(f"\n{split.upper()} | rows={len(ys):,}, V7-only RMSE={base:.5f}, "
              f"V8-only RMSE={rmse(ys,p8):.5f}")
        if mode == "full":
            expected = 478.19 if split == "long" else 452.65
            if abs(base - expected) > 0.5:
                print(f"WARNING: Reconstructed V7 Blend5 RMSE differs materially "
                      f"from the teammate's reported ~{expected:.2f}; verify run provenance.")
        for weight in WEIGHTS:
            pred = (1.0 - weight) * p7 + weight * p8
            val = rmse(ys, pred)
            print(f"  V7 {1-weight:>4.0%} + V8 Premium {weight:>3.0%}: "
                  f"RMSE={val:.5f}, delta={val-base:+.5f}")
            records.append({
                "mode": mode, "split": split, "weight_v8_premium": weight,
                "weight_v7": 1.0 - weight, "rmse": val,
                "delta_vs_v7_only": val-base,
                "n": len(ys),
            })
    OUT.mkdir(parents=True, exist_ok=True)
    destination = OUT / f"{mode}_blend_screening.csv"
    pd.DataFrame(records).to_csv(destination, index=False)
    print(f"\nSaved: {destination}")
    if mode == "quick":
        print("NOTE: quick uses TWO single-seed V7 models, NOT the original five-model V7.")
    else:
        print("NOTE: full recreates the validation-time V7 Blend5 from saved five model outputs.")
    print("These weights are diagnostic. They have NOT been tested on Kaggle.")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true", help="Only check which prediction CSVs exist")
    ap.add_argument("--mode", choices=("quick", "full"), default="quick")
    args = ap.parse_args()
    if args.check:
        check("quick")
        check("full")
    else:
        evaluate(args.mode)


if __name__ == "__main__":
    main()
