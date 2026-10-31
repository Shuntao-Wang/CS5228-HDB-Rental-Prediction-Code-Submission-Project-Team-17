"""Build one pre-specified 90% V7 Blend5 + 10% V8 Premium Kaggle candidate.

Run from the repo root (no retraining, no Kaggle upload):
    python scripts/v7_v8_make_submission.py --check
    python scripts/v7_v8_make_submission.py --write

V7: independently reproduced 5 x 3 seed original-feature blend (Kaggle public 432.97923).
V8: legal V5+8 premium feature 3 x 3 seed blend (Kaggle public 436.18827).
Fixed combination selected conservatively before looking at the Kaggle blend score:
    pred = 0.90 * pred_v7 + 0.10 * pred_v8

Full holdout diagnostic at V8 weight 10%: Long RMSE increases 0.08834,
Short RMSE decreases 0.20828. Improvement on Kaggle is NOT guaranteed.

The script validates unique non-null IDs, exact Kaggle reference-ID equality,
finite positive predictions, and output file non-overwrite protection.
It never uses unavailable Kaggle labels, suspicious external predictions,
or rounds rent to integer multiples.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from run_v2_experiments import ROOT

V7 = ROOT / "submissions/v7/blend5/submission.csv"
V8 = ROOT / "submissions/v8/premium_blend3_s3/submission.csv"
REF = ROOT / "examples/submission_reference.csv"
DEST = ROOT / "submissions/v7_v8/blend90_10/submission.csv"
V8_WEIGHT = 0.10
N = 50000


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ids_and_preds(path: Path, expected_ids: pd.Index) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"Missing required file: {path}")
    data = pd.read_csv(path, dtype={"Id": "string"})
    if data.columns.tolist() != ["Id", "Predicted"]:
        raise ValueError(f"Expected exact columns ['Id','Predicted'] in {path}")
    if len(data) != N or data["Id"].isna().any() or data["Id"].duplicated().any():
        raise ValueError(f"Expected {N} unique, non-null IDs in {path}")
    actual_ids = pd.Index(data["Id"])
    if len(actual_ids.difference(expected_ids)) or len(expected_ids.difference(actual_ids)):
        raise ValueError(f"ID set does not match reference in {path}")
    numeric = pd.to_numeric(data["Predicted"], errors="coerce")
    if numeric.isna().any():
        raise ValueError(f"Invalid non-numeric predictions in {path}")
    values = pd.Series(numeric.to_numpy(dtype=float), index=actual_ids).reindex(expected_ids).to_numpy(dtype=float)
    if len(values) != N or not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError(f"Nonfinite, nonpositive, or missing aligned predictions in {path}")
    if not np.array_equal(actual_ids.to_numpy(), expected_ids.to_numpy()):
        print(f"NOTE: {path.name} ID order differs from reference; aligned by ID.", flush=True)
    return values


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    action = ap.add_mutually_exclusive_group(required=True)
    action.add_argument("--check", action="store_true", help="Read/check sources and print diagnostics; do not save")
    action.add_argument("--write", action="store_true", help="Save the fixed 90:10 blend as a new CSV")
    args = ap.parse_args()

    ref = pd.read_csv(REF, dtype={"Id": "string"})
    if "Id" not in ref.columns or len(ref) != N or ref["Id"].isna().any() or not ref["Id"].is_unique:
        raise ValueError(f"Reference must have {N} unique, non-null IDs: {REF}")
    ids = pd.Index(ref["Id"])
    v7 = ids_and_preds(V7, ids)
    v8 = ids_and_preds(V8, ids)
    pred = (1.0 - V8_WEIGHT) * v7 + V8_WEIGHT * v8
    if not np.isfinite(pred).all() or (pred <= 0).any():
        raise ValueError("Invalid blended predictions")
    print(f"Validated: {N:,} reference IDs; original V7 and V8 predictions align.")
    print(f"V7  mean={v7.mean():.5f}, min={v7.min():.5f}, max={v7.max():.5f}")
    print(f"V8  mean={v8.mean():.5f}, min={v8.min():.5f}, max={v8.max():.5f}")
    print(f"90% V7 + 10% V8 mean={pred.mean():.5f}, min={pred.min():.5f}, max={pred.max():.5f}")
    print(f"Prediction difference (blend - V7): mean={np.mean(pred-v7):+.5f}, "
          f"MAE={np.mean(np.abs(pred-v7)):.5f}, max_abs={np.max(np.abs(pred-v7)):.5f}")
    print("Blended Kaggle Public Score: UNKNOWN until submission.")

    if args.check:
        print("CHECK PASS: no file written.")
        return

    if DEST.exists():
        old = ids_and_preds(DEST, ids)
        if not np.allclose(old, pred, rtol=0, atol=1e-7):
            raise FileExistsError(f"Destination already exists with DIFFERENT predictions. Not overwriting: {DEST}")
        print(f"Output already exists and matches: {DEST}")
        return

    DEST.parent.mkdir(parents=True, exist_ok=True)
    tmp = DEST.with_suffix(".csv.tmp")
    pd.DataFrame({"Id": ids.to_numpy(), "Predicted": pred}).to_csv(
        tmp, index=False, float_format="%.8f"
    )
    tmp.replace(DEST)
    manifest = {
        "formula": "0.90 * V7_Blend5 + 0.10 * V8_Premium",
        "weights": {"v7": 0.90, "v8_premium": 0.10},
        "input_files": {
            "v7": {"path": str(V7.relative_to(ROOT)), "sha256": sha256(V7)},
            "v8": {"path": str(V8.relative_to(ROOT)), "sha256": sha256(V8)},
        },
        "output": str(DEST.relative_to(ROOT)),
        "n_rows": N,
        "prediction_mean": float(pred.mean()),
        "holdout_full_10pct_delta_rmse": {"long": 0.08834, "short": -0.20828},
        "kaggle_score": None,
    }
    (DEST.parent / "provenance.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"SAVED: {DEST}")
    print("No Kaggle upload performed. Confirm result under My Submissions after uploading.")


if __name__ == "__main__":
    main()
