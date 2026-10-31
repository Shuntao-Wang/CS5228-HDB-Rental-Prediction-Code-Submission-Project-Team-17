"""Generate V8 Premium's 3-model x 3-seed full-train Kaggle candidate.

Paired Long/Short validation is already completed by:
    python scripts/run_v8_experiments.py --variant premium --blend --seeds 3

This script trains ONLY the full-training models, avoiding repeated holdout fits.
    python scripts/v8_make_premium_submission.py --dry-run
    python scripts/v8_make_premium_submission.py --seeds 3 --threads 4

Nine model prediction vectors are saved to disk; reruns resume interrupted training.
Uses V5 + 8 V8 Premium features, no rounding, and no leaked label or submission.
It saves a CSV locally; it does NOT upload it to Kaggle.
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from run_v2_experiments import ROOT
from run_v5_experiments import VARIANTS, fit_predict
from run_v8_experiments import attach, read_extras, BASE

MODELS = ("cat_reg", "cat_mid", "lgb_reg")
OUT = ROOT / "experiments/v8"


def fingerprint(p: Path) -> dict:
    if not p.is_file():
        raise FileNotFoundError(f"Required V8/V5 cache is missing: {p}")
    s = p.stat()
    return {"path": str(p), "size": s.st_size, "mtime_ns": s.st_mtime_ns}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if a.seeds < 1 or a.threads < 1:
        ap.error("--seeds and --threads must be positive")

    extra_path = OUT / "extras_full.pkl"
    inputs = {"v5": fingerprint(BASE), "premium_extras": fingerprint(extra_path)}
    with BASE.open("rb") as f:
        base = pickle.load(f)["full"]
    full = attach(base, read_extras("full"), "premium", "xte")
    added = [c for c in full["xtr"] if c.startswith("V8_PREMIUM_")]
    if len(added) != 8 or full["xtr"].shape[1] != base["xtr"].shape[1] + 8:
        raise AssertionError("Expected exactly 8 Premium columns on top of V5 features.")
    if (len(full["xtr"]), len(full["xte"])) != (150000, 50000):
        raise AssertionError("Unexpected full-train or test row count.")
    if full["xtr"].columns.tolist() != full["xte"].columns.tolist():
        raise AssertionError("Full-train and test feature columns are not aligned.")
    ids = pd.read_csv(ROOT / "examples/submission_reference.csv")["Id"]
    if len(ids) != 50000 or ids.isna().any() or not ids.is_unique:
        raise AssertionError("Submission IDs must be 50,000 unique, non-null values.")

    config = {
        "inputs": inputs,
        "variant": "premium",
        "models": [
            {"model": model, "lib": VARIANTS[model][0], "params": VARIANTS[model][1]}
            for model in MODELS
        ],
        "seeds": [42 + i for i in range(a.seeds)],
        "rounding": None,
        "n_features": full["xtr"].shape[1],
        "train_rows": len(full["xtr"]),
        "test_rows": len(full["xte"]),
    }
    print(f"Full dataset: {len(full['xtr']):,} training rows, {len(full['xte']):,} test rows")
    print(f"Features: V5 {base['xtr'].shape[1]} + Premium 8 = {full['xtr'].shape[1]}")
    print(f"Ensemble: {len(MODELS)} models x {a.seeds} seeds (42 through {41 + a.seeds}), unrounded")
    if a.dry_run:
        print("DRY RUN PASS: inputs and submission IDs verified; no training was performed.")
        return

    run_name = f"premium_blend3_s{a.seeds}"
    cache_dir = OUT / f"{run_name}_full"
    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest = cache_dir / "config.json"
    if manifest.exists():
        saved = json.loads(manifest.read_text(encoding="utf-8"))
        if saved != config:
            raise RuntimeError(
                f"Inputs or model settings changed since checkpoint creation: {manifest}. "
                "Do not reuse old predictions; inspect/remove this run's cache first."
            )
    else:
        manifest.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    ensemble = np.zeros(len(full["xte"]), dtype=float)
    for model in MODELS:
        lib, params = VARIANTS[model]
        for idx in range(a.seeds):
            seed = 42 + idx
            checkpoint = cache_dir / f"{model}_seed{seed}.npy"
            if checkpoint.exists():
                p = np.load(checkpoint, allow_pickle=False)
                print(f"Loaded {checkpoint.name}", flush=True)
            else:
                print(f"Training V8 Premium {model} seed={seed}...", flush=True)
                p = np.asarray(
                    fit_predict(
                        lib, params, full["xtr"], full["ytr"],
                        [full["xte"]], full["cats"], seed, a.threads
                    )[0],
                    dtype=float,
                )
                if p.shape != (len(full["xte"]),) or not np.isfinite(p).all():
                    raise AssertionError(f"Invalid predictions from {model}, seed {seed}")
                temporary = checkpoint.with_suffix(".npy.tmp")
                with temporary.open("wb") as f:
                    np.save(f, p, allow_pickle=False)
                temporary.replace(checkpoint)
                print(f"Saved {checkpoint.name}", flush=True)
            if p.shape != (len(full["xte"]),) or not np.isfinite(p).all():
                raise AssertionError(f"Invalid checkpoint: {checkpoint}")
            ensemble += p / (len(MODELS) * a.seeds)

    if not np.isfinite(ensemble).all() or (ensemble <= 0).any():
        raise AssertionError("Non-finite or nonpositive rental predictions.")
    dest = ROOT / "submissions/v8" / run_name / "submission.csv"
    dest.parent.mkdir(parents=True, exist_ok=True)
    temporary = dest.with_suffix(".csv.tmp")
    pd.DataFrame({"Id": ids.to_numpy(), "Predicted": ensemble}).to_csv(
        temporary, index=False, float_format="%.8f"
    )
    temporary.replace(dest)
    (cache_dir / "submission_metadata.json").write_text(
        json.dumps(
            {"submission_path": str(dest), "rows": len(ensemble),
             "prediction_mean": float(ensemble.mean()), "config": config},
            indent=2
        ) + "\n",
        encoding="utf-8",
    )
    print(f"Saved V8 Premium candidate: {dest}", flush=True)
    print("No Kaggle upload performed. Public leaderboard RMSE is UNKNOWN.", flush=True)


if __name__ == "__main__":
    main()
