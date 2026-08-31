"""Single shared inference path, used by both the Streamlit app and the tests.

Keeping this in one place is what guarantees a live UI prediction goes through
exactly the same transforms as training: median imputation with the *training*
medians, then standardisation with the *training* mean/std, then the model.

    raw feature dict -> imputer.transform -> scaler.transform -> model.predict_proba
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd

from data_utils import FEATURES, TARGET_NAMES, ZERO_AS_MISSING


@lru_cache(maxsize=1)
def _artifacts():
    """Load model/scaler/imputer once per process."""
    from train import load_artifacts

    return load_artifacts()


def prepare_row(values: dict) -> pd.DataFrame:
    """Turn a feature dict into a 1-row DataFrame in the model's column order.

    Any impossible zero the caller supplies is converted to NaN so the imputer
    handles it identically to a missing value in the training data.
    """
    missing = [f for f in FEATURES if f not in values]
    if missing:
        raise ValueError(f"missing features: {missing}")

    row = pd.DataFrame([{f: float(values[f]) for f in FEATURES}], columns=FEATURES)
    row[ZERO_AS_MISSING] = row[ZERO_AS_MISSING].replace(0, np.nan)
    return row


def predict_proba(values: dict) -> float:
    """Probability of diabetes (class 1) for one patient."""
    model, scaler, imputer = _artifacts()
    X = scaler.transform(imputer.transform(prepare_row(values)))
    pos_col = list(model.classes_).index(1)
    return float(model.predict_proba(X)[0, pos_col])


def predict_one(values: dict, threshold: float = 0.5) -> tuple[int, float]:
    """Return (predicted label, probability of diabetes) for one patient."""
    proba = predict_proba(values)
    return (int(proba >= threshold), proba)


def contributions(values: dict) -> pd.DataFrame:
    """Per-feature contribution to this patient's log-odds.

    contribution_i = coefficient_i * scaled_value_i, so the columns sum to
    (log-odds - intercept). Lets the UI show *why* a prediction came out as it
    did, using the model's own arithmetic rather than a separate explainer.
    """
    model, scaler, imputer = _artifacts()
    raw = prepare_row(values)
    scaled = scaler.transform(imputer.transform(raw))[0]
    coefs = np.asarray(model.coef_[0], dtype=float)

    df = pd.DataFrame(
        {
            "feature": list(FEATURES),
            "entered_value": raw.iloc[0].to_numpy(dtype=float),
            "scaled_value": scaled,
            "coefficient": coefs,
            "contribution": coefs * scaled,
        }
    )
    df["pushes_towards"] = np.where(
        df["contribution"] >= 0, TARGET_NAMES[1], TARGET_NAMES[0]
    )
    return df.sort_values("contribution", key=np.abs, ascending=False).reset_index(drop=True)


if __name__ == "__main__":
    from data_utils import load_data

    ds = load_data()
    bounds = ds.feature_bounds()

    print("=== predict.py smoke test ===")
    cases = {
        "median patient": {f: float(bounds.loc[f, "median"]) for f in FEATURES},
        "first real diabetic row": ds.X[ds.y == 1].iloc[0].fillna(
            ds.X.median()
        ).to_dict(),
        "first real non-diabetic row": ds.X[ds.y == 0].iloc[0].fillna(
            ds.X.median()
        ).to_dict(),
    }
    for name, case in cases.items():
        label, proba = predict_one(case)
        print(f"{name:28s} -> {TARGET_NAMES[label]:12s} p(diabetes)={proba:.4f}")

    print("\nContribution breakdown for the median patient:")
    print(contributions(cases["median patient"]).to_string(
        index=False, float_format=lambda v: f"{v: .4f}"))
