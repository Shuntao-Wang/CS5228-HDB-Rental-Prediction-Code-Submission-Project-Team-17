"""No-training temporal audit of the original V7 Blend5 + V8 Premium Ensemble.

Run from project root:
    python scripts/v7_v8_full_period_audit.py

Reads the five V7 s3_best validation CSVs and the V8 premium_blend3_s3
CSV (long/short). Checks row counts, month ordering, and targets, then
reports aggregate/period/month RMSE and signed bias for fixed V8 weights.

This DOES NOT train models, inspect Kaggle test labels or choose weights.
Long and Short holdouts overlap: they are different training-origin experiments.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from run_v2_experiments import ROOT
from v7_v8_prediction_blend import (
    FULL_MODELS, checked, full_path, read, rmse, v8_path,
)

OUT = ROOT / "experiments/v7_v8_blend"
WEIGHTS = (0.0, 0.1, 0.2, 0.3)
PERIODS = (
    ("2024-H1", "2024-01", "2024-06"),
    ("2024-H2-through-2025-Q1", "2024-07", "2025-03"),
)


def source(split: str):
    v8 = read(v8_path(split), "pred_v8", split)
    forecasts = []
    for model, target in FULL_MODELS:
        path = full_path(model, target, split)
        d = read(path, "pred", split)
        checked(d, v8, path)
        forecasts.append(d["pred"].to_numpy(float))
    p7 = np.stack(forecasts).mean(axis=0)
    p8 = v8["pred_v8"].to_numpy(float)
    y = v8["actual"].to_numpy(float)
    dates = pd.to_datetime(v8["month"].astype(str), errors="raise").dt.to_period("M")
    return y, p7, p8, dates


def summary_row(split, period, y, p7, p8, mask, w):
    blend = (1.0 - w) * p7[mask] + w * p8[mask]
    benchmark = rmse(y[mask], p7[mask])
    return {
        "split": split, "period": period, "n_rows": int(mask.sum()),
        "v8_weight": w,
        "rmse_v7": benchmark,
        "rmse_v8": rmse(y[mask], p8[mask]),
        "rmse_blend": rmse(y[mask], blend),
        "delta_blend_vs_v7": rmse(y[mask], blend) - benchmark,
        "bias_v7": float(np.mean(p7[mask] - y[mask])),
        "bias_v8": float(np.mean(p8[mask] - y[mask])),
        "bias_blend": float(np.mean(blend - y[mask])),
    }


def main():
    periods = []
    months = []
    for split in ("long", "short"):
        y, p7, p8, dates = source(split)
        print(f"\n===== {split.upper()} ({len(y):,} rows) =====")
        masks = [("ALL", np.ones(len(y), dtype=bool))]
        for name, start, end in PERIODS:
            mask = ((dates >= pd.Period(start)) & (dates <= pd.Period(end))).to_numpy()
            if mask.any():
                masks.append((name, mask))
        for label, mask in masks:
            print(f"\n{label}: n={int(mask.sum()):,}; V7 RMSE={rmse(y[mask],p7[mask]):.5f}; "
                  f"V8 RMSE={rmse(y[mask],p8[mask]):.5f}")
            for w in WEIGHTS:
                row = summary_row(split, label, y, p7, p8, mask, w)
                periods.append(row)
                print(f"  V8 {w:>3.0%} RMSE={row['rmse_blend']:.5f}, "
                      f"delta={row['delta_blend_vs_v7']:+.5f}, "
                      f"bias={row['bias_blend']:+.2f}")
        print("\nMonth-by-month at FIXED 20% V8 (diagnostic only):")
        for m in sorted(dates.unique()):
            mask = (dates == m).to_numpy()
            row = summary_row(split, str(m), y, p7, p8, mask, 0.2)
            months.append(row)
            print(f"  {m} | n={int(mask.sum()):,}, ΔRMSE={row['delta_blend_vs_v7']:+.4f}, "
                  f"V7 bias={row['bias_v7']:+.2f}, V8 bias={row['bias_v8']:+.2f}")
    OUT.mkdir(parents=True, exist_ok=True)
    detail = OUT / "full_period_audit.csv"
    monthly = OUT / "full_monthly_audit.csv"
    pd.DataFrame(periods).to_csv(detail, index=False, float_format="%.7f")
    pd.DataFrame(months).to_csv(monthly, index=False, float_format="%.7f")
    print(f"\nSaved: {detail}")
    print(f"Saved: {monthly}")
    print("Caution: Long and Short share validation months but use different training origins.")
    print("No weights selected, no training performed, no Kaggle test labels accessed.")


if __name__ == "__main__":
    main()
