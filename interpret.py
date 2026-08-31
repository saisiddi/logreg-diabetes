"""Interpret the fitted Logistic Regression coefficients (M7).

Because the model was trained on StandardScaler-ed features, each coefficient is
the change in the log-odds of diabetes per **one standard deviation** increase in
that feature (all else held constant). Exponentiating gives an odds ratio, which
is the form that reads naturally in prose.

    coefficient > 0  ->  odds ratio > 1  ->  raises the odds of diabetes
    coefficient < 0  ->  odds ratio < 1  ->  lowers the odds of diabetes

Coefficients on scaled features are directly comparable to each other in
magnitude, which is what makes the "top 5 most influential" ranking meaningful.

Run:  python interpret.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from data_utils import FEATURES

ROOT = Path(__file__).parent
COEFF_MD_PATH = ROOT / "coefficients.md"

TOP_N = 5

# What each feature actually measures, for readable prose.
FEATURE_BLURB = {
    "Pregnancies": "the number of times the patient has been pregnant",
    "Glucose": "plasma glucose concentration at 2 hours in an oral glucose tolerance test",
    "BloodPressure": "diastolic blood pressure (mm Hg)",
    "SkinThickness": "triceps skin-fold thickness (mm)",
    "Insulin": "2-hour serum insulin (mu U/ml)",
    "BMI": "body mass index (kg/m^2)",
    "DiabetesPedigreeFunction": "a score summarising diabetes history in the patient's relatives",
    "Age": "age in years",
}

UNITS = {
    "Pregnancies": "pregnancies",
    "Glucose": "mg/dL",
    "BloodPressure": "mm Hg",
    "SkinThickness": "mm",
    "Insulin": "mu U/ml",
    "BMI": "kg/m^2",
    "DiabetesPedigreeFunction": "pedigree units",
    "Age": "years",
}


def coefficient_table(model, feature_names: list[str] = None, scaler=None) -> pd.DataFrame:
    """Coefficients sorted by absolute value, descending.

    Columns: feature, coefficient, abs_coefficient, odds_ratio, direction and
    (when a fitted scaler is supplied) the real-world size of one SD.
    """
    feature_names = list(feature_names or FEATURES)
    coefs = np.asarray(model.coef_[0], dtype=float)

    df = pd.DataFrame(
        {
            "feature": feature_names,
            "coefficient": coefs,
            "abs_coefficient": np.abs(coefs),
            "odds_ratio": np.exp(coefs),
        }
    )
    df["direction"] = np.where(df["coefficient"] >= 0, "increases", "decreases")
    if scaler is not None:
        df["one_sd_in_raw_units"] = [
            float(scaler.scale_[feature_names.index(f)]) for f in df["feature"]
        ]
    return (
        df.sort_values("abs_coefficient", ascending=False)
        .reset_index(drop=True)
    )


def explain(row: pd.Series, rank: int) -> str:
    """One plain-English sentence about a single feature's direction of effect."""
    feat = row["feature"]
    coef = float(row["coefficient"])
    odds = float(row["odds_ratio"])
    blurb = FEATURE_BLURB.get(feat, feat)

    sd_txt = ""
    if "one_sd_in_raw_units" in row.index and pd.notna(row["one_sd_in_raw_units"]):
        sd_txt = f" (one SD is about {float(row['one_sd_in_raw_units']):.2f} {UNITS.get(feat, 'units')})"

    if coef >= 0:
        pct = (odds - 1.0) * 100.0
        return (
            f"{rank}. **{feat}** (coefficient {coef:+.4f}, odds ratio {odds:.4f}) — "
            f"{feat} is {blurb}. The coefficient is **positive**, so a "
            f"one-standard-deviation rise in {feat}{sd_txt}, with every other "
            f"feature unchanged, **increases the odds** of diabetes by about "
            f"{pct:.1f}% (odds multiplied by {odds:.4f}). Higher {feat} therefore "
            f"pushes the prediction towards diabetes."
        )
    pct = (1.0 - odds) * 100.0
    return (
        f"{rank}. **{feat}** (coefficient {coef:+.4f}, odds ratio {odds:.4f}) — "
        f"{feat} is {blurb}. The coefficient is **negative**, so a "
        f"one-standard-deviation rise in {feat}{sd_txt}, with every other feature "
        f"unchanged, **decreases the odds** of diabetes by about {pct:.1f}% "
        f"(odds multiplied by {odds:.4f}). Higher {feat} therefore pushes the "
        f"prediction away from diabetes."
    )


def top_explanations(table: pd.DataFrame, top_n: int = TOP_N) -> list[str]:
    return [explain(row, i + 1) for i, (_, row) in enumerate(table.head(top_n).iterrows())]


def table_markdown(table: pd.DataFrame) -> str:
    lines = [
        "| Rank | Feature | Coefficient (log-odds per SD) | \\|Coefficient\\| | Odds ratio e^coef | Direction of effect |",
        "|---:|---|---:|---:|---:|---|",
    ]
    for i, (_, r) in enumerate(table.iterrows(), start=1):
        lines.append(
            f"| {i} | {r['feature']} | {r['coefficient']:+.4f} | "
            f"{r['abs_coefficient']:.4f} | {r['odds_ratio']:.4f} | "
            f"{r['direction']} odds of diabetes |"
        )
    return "\n".join(lines)


def build_markdown(model, table: pd.DataFrame, metrics: dict | None = None) -> str:
    intercept = float(model.intercept_[0])
    base_odds = float(np.exp(intercept))
    base_prob = base_odds / (1.0 + base_odds)

    parts = [
        "# Logistic Regression coefficients — diabetes classifier",
        "",
        "Every number on this page is read straight off the fitted",
        "`LogisticRegression` in `model.pkl` (`model.coef_`, `model.intercept_`).",
        "Regenerate with `python interpret.py`.",
        "",
        "## How to read these numbers",
        "",
        "The model was trained on **standardised** features, so each coefficient is",
        "the change in the **log-odds** of diabetes caused by a **one-standard-deviation**",
        "increase in that feature, holding the other seven fixed. `exp(coefficient)` is",
        "the **odds ratio**: multiply the current odds by it to get the new odds.",
        "",
        "- positive coefficient → odds ratio above 1 → **increases** the odds of diabetes",
        "- negative coefficient → odds ratio below 1 → **decreases** the odds of diabetes",
        "",
        "Because the inputs share a common scale, the magnitudes are comparable, so",
        "sorting by |coefficient| is a fair influence ranking.",
        "",
        f"**Intercept:** {intercept:+.4f} — the log-odds for a patient sitting exactly at",
        f"the training mean on all eight features. That is odds of {base_odds:.4f}, i.e. a",
        f"predicted probability of diabetes of {base_prob:.4f} for an average patient,",
        "which tracks the dataset's 34.9% positive rate.",
        "",
        "## Full coefficient table (sorted by |coefficient|, descending)",
        "",
        table_markdown(table),
        "",
        f"## Plain-English reading of the top {TOP_N} features",
        "",
    ]
    parts.extend(s + "\n" for s in top_explanations(table))

    weakest = table.iloc[-1]
    parts += [
        "## The rest",
        "",
        f"The remaining features carry smaller weights. The weakest is "
        f"**{weakest['feature']}** at {weakest['coefficient']:+.4f} "
        f"(odds ratio {weakest['odds_ratio']:.4f}), close enough to zero that it barely "
        "moves a prediction once the stronger features are accounted for.",
        "",
        "## Caveats",
        "",
        "- These are **associations in this dataset, not causal effects**. A positive",
        "  coefficient does not mean changing the feature would change a patient's risk.",
        "- The five zero-coded columns were median-imputed (see `PROGRESS_LOG.md` M2).",
        "  `Insulin` was missing for 48.7% of rows and `SkinThickness` for 29.6%, so",
        "  their coefficients are estimated on heavily reconstructed data and should be",
        "  trusted less than the others.",
        "- Coefficients are correlation-sensitive: `BMI` and `SkinThickness` are strongly",
        "  related, so the model splits the shared signal between them somewhat",
        "  arbitrarily.",
        "- This is a teaching model, not a diagnostic device.",
    ]

    if metrics:
        parts += [
            "",
            "## Model these coefficients came from",
            "",
            f"- Test-set accuracy: {metrics['accuracy']:.4f}",
            f"- Test-set ROC-AUC: {metrics['roc_auc']:.4f}",
            f"- Test-set F1 (diabetes): {metrics['f1_score']:.4f}",
            f"- Evaluated on n = {metrics['n_test']} held-out patients",
        ]

    return "\n".join(parts) + "\n"


def print_table(table: pd.DataFrame, model) -> None:
    print("=== M7: coefficient interpretation ===")
    print(f"Intercept: {float(model.intercept_[0]):+.6f}\n")
    cols = ["feature", "coefficient", "abs_coefficient", "odds_ratio", "direction"]
    if "one_sd_in_raw_units" in table.columns:
        cols.append("one_sd_in_raw_units")
    print(table[cols].to_string(index=False, float_format=lambda v: f"{v: .4f}"))
    print(f"\n--- plain-English top {TOP_N} ---")
    for s in top_explanations(table):
        print("\n" + s.replace("**", ""))


if __name__ == "__main__":
    import joblib

    from evaluate import METRICS_PATH, load_metrics
    from train import MODEL_PATH, SCALER_PATH

    if not MODEL_PATH.exists():
        raise SystemExit("model.pkl not found. Run `python train.py` first.")

    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    table = coefficient_table(model, FEATURES, scaler=scaler)
    metrics = load_metrics() if METRICS_PATH.exists() else None

    print_table(table, model)

    COEFF_MD_PATH.write_text(build_markdown(model, table, metrics), encoding="utf-8")
    print(f"\nSaved -> {COEFF_MD_PATH.name}")
