"""Split, impute, scale and train the Logistic Regression diabetes classifier.

Pipeline order (and why it matters)
-----------------------------------
    split  ->  impute (train medians)  ->  scale (train mean/std)  ->  fit

Both the imputer and the scaler are fitted on the TRAINING split only and then
applied unchanged to the test split, so no test-set statistic ever leaks into
training. Three artifacts are persisted so that `evaluate.py`, `interpret.py`
and the Streamlit app all reuse the exact same transforms:

    imputer.pkl   SimpleImputer(strategy="median")   <- extra artifact, see M2
    scaler.pkl    StandardScaler()
    model.pkl     LogisticRegression(max_iter=1000)

Run:  python train.py
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from data_utils import (
    FEATURES,
    RANDOM_STATE,
    TARGET_NAMES,
    Dataset,
    load_data,
    split_data,
)

ROOT = Path(__file__).parent
MODEL_PATH = ROOT / "model.pkl"
SCALER_PATH = ROOT / "scaler.pkl"
IMPUTER_PATH = ROOT / "imputer.pkl"
TRAIN_INFO_PATH = ROOT / "train_info.json"

MAX_ITER = 1000


def build_preprocessors(
    X_train: pd.DataFrame, X_test: pd.DataFrame
) -> tuple[SimpleImputer, StandardScaler, np.ndarray, np.ndarray]:
    """Fit imputer + scaler on train only; return them and both transformed splits."""
    imputer = SimpleImputer(strategy="median")
    Xtr_i = imputer.fit_transform(X_train)
    Xte_i = imputer.transform(X_test)  # train medians, not test medians

    scaler = StandardScaler()
    Xtr_s = scaler.fit_transform(Xtr_i)
    Xte_s = scaler.transform(Xte_i)  # train mean/std, not test mean/std

    return imputer, scaler, Xtr_s, Xte_s


def scaling_check(Xtr_s: np.ndarray, feature_names: list[str]) -> pd.DataFrame:
    """M4 evidence: post-scaling per-feature mean and std of the training split."""
    return pd.DataFrame(
        {
            "mean_after_scaling": Xtr_s.mean(axis=0),
            "std_after_scaling": Xtr_s.std(axis=0),
        },
        index=feature_names,
    )


def train(dataset: Dataset | None = None, save: bool = True):
    """Run M3->M5 and return everything downstream milestones need."""
    dataset = dataset or load_data()
    X_train, X_test, y_train, y_test = split_data(dataset)

    imputer, scaler, Xtr_s, Xte_s = build_preprocessors(X_train, X_test)

    model = LogisticRegression(max_iter=MAX_ITER, random_state=RANDOM_STATE)
    model.fit(Xtr_s, y_train)

    artifacts = {
        "model": model,
        "scaler": scaler,
        "imputer": imputer,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "X_train_scaled": Xtr_s,
        "X_test_scaled": Xte_s,
        "dataset": dataset,
    }

    if save:
        joblib.dump(model, MODEL_PATH)
        joblib.dump(scaler, SCALER_PATH)
        joblib.dump(imputer, IMPUTER_PATH)
        TRAIN_INFO_PATH.write_text(
            json.dumps(
                {
                    "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "data_source": dataset.source,
                    "is_synthetic": dataset.is_synthetic,
                    "n_train": int(len(X_train)),
                    "n_test": int(len(X_test)),
                    "max_iter": MAX_ITER,
                    "n_iter_": int(model.n_iter_[0]),
                    "solver": model.solver,
                    "random_state": RANDOM_STATE,
                    "feature_names": list(FEATURES),
                    "imputer_medians": {
                        f: float(v) for f, v in zip(FEATURES, imputer.statistics_)
                    },
                    "scaler_mean": {f: float(v) for f, v in zip(FEATURES, scaler.mean_)},
                    "scaler_scale": {f: float(v) for f, v in zip(FEATURES, scaler.scale_)},
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    return artifacts


def load_artifacts():
    """Load the three persisted transforms. Trains first if any are missing."""
    if not (MODEL_PATH.exists() and SCALER_PATH.exists() and IMPUTER_PATH.exists()):
        train()
    return (
        joblib.load(MODEL_PATH),
        joblib.load(SCALER_PATH),
        joblib.load(IMPUTER_PATH),
    )


if __name__ == "__main__":
    ds = load_data()
    if ds.is_synthetic:
        print("[train] WARNING training on SYNTHETIC fallback data.\n")

    a = train(ds)
    X_train, X_test = a["X_train"], a["X_test"]
    y_train, y_test = a["y_train"], a["y_test"]

    print("=== M3: 80/20 stratified split (random_state=42) ===")
    print(f"Train : {X_train.shape[0]} rows   Test : {X_test.shape[0]} rows "
          f"(test fraction {X_test.shape[0] / len(ds.df):.4f})")
    for name, y in [("full ", ds.y), ("train", y_train), ("test ", y_test)]:
        counts = y.value_counts().sort_index()
        bal = {TARGET_NAMES[int(i)]: int(c) for i, c in counts.items()}
        print(f"  {name} class balance: {bal}  positive rate = {y.mean():.4f}")

    print("\n=== M4: StandardScaler fitted on TRAIN only ===")
    print("Median values learned from train and used to fill the impossible zeros:")
    print(pd.Series(a["imputer"].statistics_, index=FEATURES).to_string())
    check = scaling_check(a["X_train_scaled"], list(FEATURES))
    print("\nTraining split after scaling (must be mean~0, std~1):")
    print(check.to_string(float_format=lambda v: f"{v: .6e}"))
    print(f"\nmax |mean|  = {np.abs(check['mean_after_scaling']).max():.3e}  (target 0)")
    print(f"max |std-1| = {np.abs(check['std_after_scaling'] - 1).max():.3e}  (target 0)")
    print("\nTest split after transform with the TRAIN scaler "
          "(mean/std deliberately NOT 0/1):")
    print(
        pd.DataFrame(
            {
                "mean": a["X_test_scaled"].mean(axis=0),
                "std": a["X_test_scaled"].std(axis=0),
            },
            index=FEATURES,
        ).to_string(float_format=lambda v: f"{v: .4f}")
    )
    print(f"NaNs remaining after imputation: train="
          f"{int(np.isnan(a['X_train_scaled']).sum())} "
          f"test={int(np.isnan(a['X_test_scaled']).sum())}")

    print("\n=== M5: LogisticRegression training ===")
    model = a["model"]
    print(f"Estimator     : {model}")
    print(f"Solver        : {model.solver}")
    print(f"max_iter      : {model.max_iter}")
    print(f"n_iter_ used  : {int(model.n_iter_[0])} "
          f"({'converged' if model.n_iter_[0] < model.max_iter else 'DID NOT CONVERGE'})")
    print(f"coef_ shape   : {model.coef_.shape}")
    print(f"intercept_    : {float(model.intercept_[0]):+.6f}")
    print(f"Train accuracy: {model.score(a['X_train_scaled'], y_train):.4f}")
    print(f"\nSaved -> {MODEL_PATH.name}, {SCALER_PATH.name}, {IMPUTER_PATH.name}, "
          f"{TRAIN_INFO_PATH.name}")
