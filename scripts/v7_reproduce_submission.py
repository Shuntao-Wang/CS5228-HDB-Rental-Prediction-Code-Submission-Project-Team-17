"""Reproduce the teammate's original V7 Blend5 test prediction, without rerunning holdouts.

Use from repository root on branch feature/v7-v8-premium-fusion-20261029:
    python scripts/v7_reproduce_submission.py --check
    python scripts/v7_reproduce_submission.py --model cat_reg_ratio
    python scripts/v7_reproduce_submission.py --model cat_mid_ratio
    python scripts/v7_reproduce_submission.py --model lgb_reg_ratio
    python scripts/v7_reproduce_submission.py --model cat_reg_raw
    python scripts/v7_reproduce_submission.py --model cat_mid_raw
    python scripts/v7_reproduce_submission.py --blend

The per-model prediction is produced by the original scripts/run_v7_experiments.py
run_fold() function, using exactly the validation-reproduced configuration:
--variant, --target, 3 seeds (42/43/44), --drift 0.5, no dropped columns,
the original VARIANTS hyperparameters, and the exact V7 cache.
Full train is 150,000 rows, test is 50,000 rows, 69 features.

This workflow does NOT refit holdouts or overwrite any existing V7 run.
Each completed model is saved independently to permit safe incremental runs.
Inputs are Kaggle-provided features/labels only, with strict pre-month history.
No external same-period rental labels or prior leaked submissions are used.
No automated Kaggle upload or leaderboard tuning is performed.
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from run_v2_experiments import ROOT
from run_v5_experiments import VARIANTS
from run_v7_experiments import run_fold

CACHE = ROOT / "experiments/v7/fc_v5s3nod.pkl"
OUT = ROOT / "experiments/v7/repro_full"
SUBMISSION = ROOT / "submissions/v7/blend5/submission.csv"
REFERENCE = ROOT / "examples/submission_reference.csv"
MODELS = (
    ("cat_reg", "ratio"),
    ("cat_mid", "ratio"),
    ("lgb_reg", "ratio"),
    ("cat_reg", "raw"),
    ("cat_mid", "raw"),
)
SEEDS = 3
DRIFT = 0.5
THREADS_DEFAULT = 4


def name(model: str, target: str) -> str:
    return f"{model}_{target}"


def pred_file(model: str, target: str) -> Path:
    return OUT / name(model, target) / "test_pred.csv"


def metadata_file(model: str, target: str) -> Path:
    return OUT / name(model, target) / "config.json"


def file_fingerprint(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Required input is missing: {path}")
    s = path.stat()
    return {"size_bytes": s.st_size, "modified_ns": s.st_mtime_ns}


def expected_config(model: str, target: str) -> dict:
    lib, params = VARIANTS[model]
    return {
        "feature_cache": str(CACHE.relative_to(ROOT)),
        "feature_cache_fingerprint": file_fingerprint(CACHE),
        "model": model,
        "target": target,
        "library": lib,
        "parameters": params,
        "seeds": [42, 43, 44],
        "drift": DRIFT,
        "drop": [],
        "train_rows": 150000,
        "test_rows": 50000,
        "features": 69,
    }


def reference_ids() -> pd.Series:
    d = pd.read_csv(REFERENCE)
    if "Id" not in d or len(d) != 50000 or d["Id"].isna().any() or not d["Id"].is_unique:
        raise ValueError("Submission reference must have exactly 50,000 unique, non-null Id values.")
    return d["Id"]


def read_model_pred(model: str, target: str, ids: pd.Series) -> np.ndarray:
    file = pred_file(model, target)
    meta = metadata_file(model, target)
    if not file.is_file() or not meta.is_file():
        raise FileNotFoundError(f"Missing model output or manifest: {file} / {meta}")
    actual_config = json.loads(meta.read_text(encoding="utf-8"))
    if actual_config != expected_config(model, target):
        raise ValueError(f"Model metadata/cache fingerprint mismatch: {meta}")
    df = pd.read_csv(file)
    if list(df.columns) != ["Id", "Predicted"] or len(df) != len(ids):
        raise ValueError(f"Incorrect submission shape/columns: {file}")
    if not np.array_equal(df["Id"].to_numpy(), ids.to_numpy()):
        raise ValueError(f"Test Id order mismatch: {file}")
    pred = pd.to_numeric(df["Predicted"], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(pred).all() or (pred <= 0).any():
        raise ValueError(f"Invalid or nonpositive prediction: {file}")
    return pred


def check() -> None:
    print(f"V7 cache: {CACHE.relative_to(ROOT)}")
    print(f"Cache fingerprint: {file_fingerprint(CACHE)}")
    ids = reference_ids()
    print(f"Submission reference: {len(ids):,} unique IDs")
    with CACHE.open("rb") as f:
        cache = pickle.load(f)
    if "full" not in cache:
        raise ValueError("V7 cache lacks the full train/test split.")
    c = cache["full"]
    tr, te = c["xtr"], c["xte"]
    if (len(tr), len(te), tr.shape[1], te.shape[1]) != (150000, 50000, 69, 69):
        raise ValueError(f"Unexpected V7 full shapes: train={tr.shape}, test={te.shape}")
    if tr.columns.tolist() != te.columns.tolist():
        raise ValueError("Full-train and test feature columns disagree.")
    if len(c["ytr"]) != len(tr) or len(c["mtr"]) != len(tr):
        raise ValueError("Invalid training labels/months in V7 full cache.")
    if len(c["lte"]) != len(te) or len(c["mte"]) != len(te):
        raise ValueError("Invalid test market-level inputs.")
    if not np.isfinite(c["ytr"]).all() or not np.isfinite(c["lte"]).all() or (c["lte"] <= 0).any():
        raise ValueError("Invalid V7 full labels or forecast market levels.")
    print(f"FULL DATA READY: train={len(tr):,}, test={len(te):,}, features={tr.shape[1]}")
    for model, target in MODELS:
        try:
            p = read_model_pred(model, target, ids)
            print(f"  COMPLETE {name(model,target):15s}  mean={p.mean():.2f}  {pred_file(model,target).relative_to(ROOT)}")
        except FileNotFoundError:
            print(f"  MISSING  {name(model,target):15s}  (not trained yet)")
        except ValueError as e:
            print(f"  INVALID  {name(model,target):15s}  {e}")
            raise


def run_one(choice: str, threads: int) -> None:
    model, target = next(pair for pair in MODELS if name(*pair) == choice)
    ids = reference_ids()
    target_file = pred_file(model, target)
    config_path = metadata_file(model, target)
    config = expected_config(model, target)
    if target_file.exists() or config_path.exists():
        if target_file.exists() and config_path.exists():
            existing = read_model_pred(model, target, ids)
            print(f"ALREADY COMPLETE: {choice}; n={len(existing):,}, mean={existing.mean():.2f}")
            return
        raise FileExistsError(f"Partial output already exists in {target_file.parent}; inspect before proceeding.")

    with CACHE.open("rb") as f:
        cache = pickle.load(f)
    c = cache["full"]
    if (len(c["xtr"]), len(c["xte"]), c["xtr"].shape[1]) != (150000, 50000, 69):
        raise ValueError("Unexpected full V7 train/test shape.")
    if c["xtr"].columns.tolist() != c["xte"].columns.tolist():
        raise ValueError("Train/test feature column mismatch.")
    lib, params = VARIANTS[model]
    print(f"Training original V7 {choice}, full 150,000 -> 50,000, "
          f"{SEEDS} seeds (42/43/44), drift={DRIFT}, threads={threads}...", flush=True)
    pred, nfeat = run_fold(lib, params, c, "xte", "lte", target,
                           SEEDS, threads, [], DRIFT)
    pred = np.asarray(pred, dtype=float)
    if nfeat != 69 or pred.shape != (50000,) or not np.isfinite(pred).all() or (pred <= 0).any():
        raise ValueError("Invalid full V7 model output.")
    target_file.parent.mkdir(parents=True, exist_ok=True)
    temporary = target_file.with_suffix(".csv.tmp")
    pd.DataFrame({"Id": ids.to_numpy(), "Predicted": pred}).to_csv(
        temporary, index=False, float_format="%.8f"
    )
    temporary.replace(target_file)
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    print(f"SAVED {choice}: {target_file.relative_to(ROOT)}")
    print(f"  rows={len(pred):,}, mean={pred.mean():.3f}, min={pred.min():.3f}, max={pred.max():.3f}")


def blend() -> None:
    ids = reference_ids()
    preds = []
    for model, target in MODELS:
        pred = read_model_pred(model, target, ids)
        print(f"  {name(model,target):15s} mean={pred.mean():.3f}")
        preds.append(pred)
    # Equal weight ACROSS five 3-seed model families, as in teammate V7 Blend5.
    ensemble = np.mean(np.stack(preds), axis=0)
    if not np.isfinite(ensemble).all() or (ensemble <= 0).any():
        raise ValueError("Invalid V7 Blend5 predictions.")
    if SUBMISSION.exists():
        current = pd.read_csv(SUBMISSION)
        if list(current.columns) != ["Id", "Predicted"] or len(current) != len(ids):
            raise FileExistsError(f"Existing file differs or cannot be verified: {SUBMISSION}")
        values = pd.to_numeric(current["Predicted"], errors="coerce").to_numpy(float)
        if not np.array_equal(current["Id"].to_numpy(), ids.to_numpy()) or not np.allclose(
            values, ensemble, rtol=0, atol=1e-5
        ):
            raise FileExistsError(f"Submission exists with different values; not overwriting: {SUBMISSION}")
        print(f"ALREADY EXISTS AND MATCHES: {SUBMISSION}")
        return
    SUBMISSION.parent.mkdir(parents=True, exist_ok=True)
    tmp = SUBMISSION.with_suffix(".csv.tmp")
    pd.DataFrame({"Id": ids.to_numpy(), "Predicted": ensemble}).to_csv(
        tmp, index=False, float_format="%.8f"
    )
    tmp.replace(SUBMISSION)
    print(f"\nSAVED V7 Blend5: {SUBMISSION}")
    print(f"Rows={len(ensemble):,}, mean={ensemble.mean():.3f}")
    print("The Kaggle score is not known until uploaded; no upload was performed.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--check", action="store_true")
    action.add_argument("--model", choices=[name(*m) for m in MODELS])
    action.add_argument("--blend", action="store_true")
    parser.add_argument("--threads", type=int, default=THREADS_DEFAULT)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    if args.check:
        check()
    elif args.model is not None:
        run_one(args.model, args.threads)
    else:
        blend()


if __name__ == "__main__":
    main()
