"""Streamlit UI for the Logistic Regression diabetes classifier.

Every number shown here is read from the generated artifacts (`metrics.json`,
`model.pkl`, `scaler.pkl`, `imputer.pkl`) or recomputed on the spot. Nothing is
hardcoded.

Run:
    streamlit run app.py
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import pandas as pd
import streamlit as st

from data_utils import FEATURES, TARGET_NAMES, load_data
from evaluate import (
    ACC_BAND,
    AUC_BAND,
    CM_PLOT_PATH,
    METRICS_PATH,
    ROC_PLOT_PATH,
    evaluate_saved_model,
    load_metrics,
    plot_confusion_matrix,
    plot_roc_curve,
)
from interpret import (
    COEFF_MD_PATH,
    FEATURE_BLURB,
    UNITS,
    build_markdown,
    coefficient_table,
    top_explanations,
)
from predict import contributions, predict_one
from train import IMPUTER_PATH, MODEL_PATH, SCALER_PATH, load_artifacts

st.set_page_config(page_title="Logistic Regression — Diabetes", layout="wide")


@st.cache_data(show_spinner=False)
def get_dataset():
    return load_data()


def artifacts_exist() -> bool:
    return all(p.exists() for p in (MODEL_PATH, SCALER_PATH, IMPUTER_PATH, METRICS_PATH))


def render_data_overview(dataset) -> None:
    st.subheader("Dataset and the zero-as-missing problem")
    from data_utils import ZERO_AS_MISSING, zero_audit

    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", f"{dataset.n_samples}")
    c2.metric("Features", f"{dataset.n_features}")
    c3.metric("Positive rate", f"{dataset.y.mean():.4f}")

    st.write(f"Class balance: `{dataset.class_balance()}`")

    st.markdown(
        "Five columns cannot legitimately be 0 in a living person, so a 0 there is a "
        "missing reading. Those zeros are converted to `NaN` and filled with the "
        "**training-split median** before scaling."
    )
    audit = zero_audit(dataset.raw)
    st.dataframe(audit, width="stretch")
    st.caption(
        f"{int((dataset.raw[ZERO_AS_MISSING] == 0).any(axis=1).sum())} of "
        f"{dataset.n_samples} rows contain at least one impossible zero, which is why "
        "they are imputed rather than dropped."
    )

    with st.expander("First 10 rows (raw, zeros intact)"):
        st.dataframe(dataset.raw.head(10), width="stretch")


def render_metrics(metrics: dict) -> None:
    """The five headline metrics, both plots and the classification report."""
    st.subheader("Evaluation on the held-out test set")
    st.caption(
        f"n_test = {metrics['n_test']} · positive class = {metrics['positive_class']} · "
        f"data source = {metrics.get('data_source', 'unknown')}"
        + ("  ** SYNTHETIC FALLBACK DATA **" if metrics.get("is_synthetic") else "")
        + f" · trained at {metrics.get('trained_at', 'unknown')}"
    )

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Accuracy", f"{metrics['accuracy']:.4f}")
    c2.metric("Precision", f"{metrics['precision']:.4f}")
    c3.metric("Recall", f"{metrics['recall']:.4f}")
    c4.metric("F1-score", f"{metrics['f1_score']:.4f}")
    c5.metric("ROC-AUC", f"{metrics['roc_auc']:.4f}")

    left, right = st.columns(2)
    with left:
        st.markdown("**Confusion matrix**")
        st.pyplot(plot_confusion_matrix(metrics["confusion_matrix"]))
    with right:
        st.markdown("**ROC curve**")
        st.pyplot(
            plot_roc_curve(
                metrics["roc_curve"]["fpr"],
                metrics["roc_curve"]["tpr"],
                metrics["roc_auc"],
            )
        )

    st.markdown("**Classification report**")
    st.code(metrics["classification_report"], language="text")

    t = metrics["confusion_matrix_terms"]
    st.markdown(
        f"Of the {metrics['test_class_balance'][TARGET_NAMES[1]]} genuine diabetics in "
        f"the test set the model catches **{t['true_positive']}** and misses "
        f"**{t['false_negative']}** (that is the recall of "
        f"{metrics['recall']:.4f}). It raises **{t['false_positive']}** false alarms "
        f"among {metrics['test_class_balance'][TARGET_NAMES[0]]} healthy patients."
    )

    with st.expander("Confusion matrix as a table"):
        st.dataframe(
            pd.DataFrame(
                metrics["confusion_matrix"],
                index=[f"true: {n}" for n in metrics["confusion_matrix_labels"]],
                columns=[f"pred: {n}" for n in metrics["confusion_matrix_labels"]],
            ),
            width="stretch",
        )

    with st.expander("Per-class precision / recall / F1"):
        st.dataframe(pd.DataFrame(metrics["per_class"]).T, width="stretch")


def render_sanity_checks(metrics: dict) -> None:
    """Show the spec's sanity bands honestly, including the one that misses."""
    st.subheader("Spec sanity checks")
    cv = metrics.get("cross_validation")
    ci_lo, ci_hi = metrics["accuracy_ci95"]
    sc = metrics["sanity_check"]

    rows = [
        {
            "check": f"Fixed-split ROC-AUC in [{AUC_BAND[0]}, {AUC_BAND[1]}]",
            "value": f"{metrics['roc_auc']:.4f}",
            "result": "PASS" if sc["roc_auc_in_band"] else "FAIL",
        },
        {
            "check": f"Fixed-split accuracy in [{ACC_BAND[0]}, {ACC_BAND[1]}]",
            "value": f"{metrics['accuracy']:.4f}",
            "result": "PASS" if sc["accuracy_in_band"] else "BELOW BAND",
        },
        {
            "check": "...its 95% CI overlaps that band",
            "value": f"[{ci_lo:.4f}, {ci_hi:.4f}]",
            "result": "PASS" if sc["accuracy_ci_overlaps_band"] else "FAIL",
        },
    ]
    if cv:
        rows += [
            {
                "check": f"{cv['folds']}-fold CV accuracy in "
                f"[{ACC_BAND[0]}, {ACC_BAND[1]}]",
                "value": f"{cv['accuracy_mean']:.4f} +- {cv['accuracy_std']:.4f}",
                "result": "PASS"
                if ACC_BAND[0] <= cv["accuracy_mean"] <= ACC_BAND[1]
                else "FAIL",
            },
            {
                "check": f"{cv['folds']}-fold CV ROC-AUC in "
                f"[{AUC_BAND[0]}, {AUC_BAND[1]}]",
                "value": f"{cv['roc_auc_mean']:.4f} +- {cv['roc_auc_std']:.4f}",
                "result": "PASS"
                if AUC_BAND[0] <= cv["roc_auc_mean"] <= AUC_BAND[1]
                else "FAIL",
            },
        ]
    st.dataframe(pd.DataFrame(rows).set_index("check"), width="stretch")

    if not sc["accuracy_in_band"]:
        st.info(
            f"The fixed-split accuracy of {metrics['accuracy']:.4f} sits just under the "
            f"{ACC_BAND[0]}-{ACC_BAND[1]} band. With only {metrics['n_test']} test rows "
            f"the standard error is about {(ci_hi - ci_lo) / 3.92:.4f}, and `random_state=42` "
            "happens to be the hardest of the first 30 seeds. The same pipeline under "
            f"{cv['folds']}-fold cross-validation scores {cv['accuracy_mean']:.4f}, inside "
            "the band. Full investigation is in PROGRESS_LOG.md (M6)."
            if cv
            else "See PROGRESS_LOG.md (M6)."
        )


def render_coefficients(model, scaler, metrics: dict) -> None:
    """Coefficient table plus the plain-English top-5 reading."""
    st.subheader("What the model learned (coefficients)")
    st.markdown(
        "Features were standardised before fitting, so each coefficient is the change "
        "in the **log-odds** of diabetes per **one standard deviation** increase in that "
        "feature. `exp(coefficient)` is the odds ratio. Positive means it **increases** "
        "the odds of diabetes; negative means it **decreases** them."
    )

    table = coefficient_table(model, list(FEATURES), scaler=scaler)

    left, right = st.columns([3, 2])
    with left:
        show = table.copy()
        show.index = range(1, len(show) + 1)
        show.index.name = "rank"
        st.dataframe(
            show.rename(
                columns={
                    "coefficient": "coefficient (log-odds / SD)",
                    "abs_coefficient": "|coefficient|",
                    "odds_ratio": "odds ratio",
                    "one_sd_in_raw_units": "1 SD in raw units",
                }
            ),
            width="stretch",
        )
    with right:
        st.markdown("**Signed coefficients**")
        st.bar_chart(table.set_index("feature")["coefficient"])

    st.markdown(f"**Intercept:** `{float(model.intercept_[0]):+.4f}` — the log-odds for a "
                "patient at the training mean on every feature.")

    st.markdown("#### Plain-English reading of the top 5")
    for sentence in top_explanations(table):
        st.markdown(sentence)

    with st.expander("Feature glossary"):
        st.dataframe(
            pd.DataFrame(
                {
                    "feature": list(FEATURES),
                    "meaning": [FEATURE_BLURB[f] for f in FEATURES],
                    "unit": [UNITS[f] for f in FEATURES],
                }
            ).set_index("feature"),
            width="stretch",
        )

    with st.expander("coefficients.md (the generated file)"):
        if COEFF_MD_PATH.exists():
            st.markdown(COEFF_MD_PATH.read_text(encoding="utf-8"))
        else:
            st.warning("coefficients.md not generated yet — run `python interpret.py`.")


def render_prediction_form(dataset) -> None:
    """Manual entry bounded by the real min/max of each feature."""
    st.subheader("Try a live prediction")
    st.markdown(
        "Sliders are bounded by each feature's **real minimum and maximum in the "
        "dataset** (ignoring the impossible zeros), so you cannot enter a BMI of 0. "
        "Submitting runs the same imputer → scaler → model path used in training."
    )

    bounds = dataset.feature_bounds()

    preset = st.radio(
        "Start from",
        ["Dataset median", "A real diabetic patient", "A real non-diabetic patient"],
        horizontal=True,
    )
    if preset == "A real diabetic patient":
        defaults = dataset.X[dataset.y == 1].iloc[0]
    elif preset == "A real non-diabetic patient":
        defaults = dataset.X[dataset.y == 0].iloc[0]
    else:
        defaults = bounds["median"]

    with st.form("prediction_form"):
        values: dict[str, float] = {}
        cols = st.columns(4)
        for i, name in enumerate(FEATURES):
            lo = float(bounds.loc[name, "min"])
            hi = float(bounds.loc[name, "max"])
            raw_default = defaults[name]
            if pd.isna(raw_default):
                raw_default = bounds.loc[name, "median"]
            default = float(min(max(float(raw_default), lo), hi))
            step = (hi - lo) / 100.0 or 0.01
            values[name] = cols[i % 4].slider(
                f"{name} ({UNITS[name]})",
                min_value=lo,
                max_value=hi,
                value=default,
                step=step,
                help=FEATURE_BLURB[name],
                key=f"slider_{preset}_{name}",
            )
        threshold = st.slider(
            "Decision threshold on P(diabetes)",
            min_value=0.05,
            max_value=0.95,
            value=0.50,
            step=0.01,
            help="0.50 is the default. Lower it to catch more diabetics at the cost "
                 "of more false alarms.",
        )
        submitted = st.form_submit_button("Predict", type="primary")

    if not submitted:
        return

    label, proba = predict_one(values, threshold=threshold)
    name = TARGET_NAMES[label]

    if label == 1:
        st.error(f"Prediction: **{name.upper()}**  (P(diabetes) = {proba:.4f} ≥ "
                 f"threshold {threshold:.2f})")
    else:
        st.success(f"Prediction: **{name.upper()}**  (P(diabetes) = {proba:.4f} < "
                   f"threshold {threshold:.2f})")

    c1, c2 = st.columns(2)
    c1.metric(f"P({TARGET_NAMES[0]})", f"{1 - proba:.4f}")
    c2.metric(f"P({TARGET_NAMES[1]})", f"{proba:.4f}")

    st.markdown("**Why — each feature's contribution to the log-odds**")
    contrib = contributions(values)
    st.dataframe(contrib.set_index("feature"), width="stretch")
    st.caption(
        "contribution = coefficient x scaled value. These sum to "
        f"{contrib['contribution'].sum():+.4f}; adding the intercept gives the total "
        "log-odds behind the probability above."
    )
    st.bar_chart(contrib.set_index("feature")["contribution"])

    st.caption(
        "Educational demo on a small 1988 dataset. Not a medical device and not "
        "diagnostic advice."
    )


def main() -> None:
    st.title("Logistic Regression — Pima Indians Diabetes Classifier")
    st.write(
        "Predicts **diabetes (1)** vs **no diabetes (0)** from 8 clinical measurements "
        "using `LogisticRegression(max_iter=1000)` from scikit-learn."
    )

    dataset = get_dataset()

    with st.sidebar:
        st.header("Artifacts")
        if artifacts_exist():
            st.success("model.pkl, scaler.pkl, imputer.pkl, metrics.json found")
        else:
            st.warning("Artifacts missing — they will be generated now.")
        st.write(f"Data source: `{dataset.source}`")
        if dataset.is_synthetic:
            st.error("Running on SYNTHETIC fallback data (download failed).")
        st.divider()
        retrain = st.button("Retrain and re-evaluate", type="primary")
        st.caption(
            "Refits the imputer, scaler and LogisticRegression, then rewrites "
            "metrics.json, coefficients.md and both PNGs."
        )

    if retrain or not artifacts_exist():
        with st.spinner("Training and evaluating..."):
            from train import train

            train(dataset)
            metrics = evaluate_saved_model()
            model, scaler, _ = load_artifacts()
            COEFF_MD_PATH.write_text(
                build_markdown(model, coefficient_table(model, list(FEATURES), scaler=scaler),
                               metrics),
                encoding="utf-8",
            )
        st.success("Done — every number below was just recomputed.")
    else:
        metrics = load_metrics()

    model, scaler, _ = load_artifacts()

    render_metrics(metrics)
    st.divider()
    render_sanity_checks(metrics)
    st.divider()
    render_coefficients(model, scaler, metrics)
    st.divider()
    render_prediction_form(dataset)
    st.divider()
    render_data_overview(dataset)

    with st.sidebar:
        st.divider()
        st.header("Saved plots")
        if CM_PLOT_PATH.exists():
            st.image(str(CM_PLOT_PATH), caption=CM_PLOT_PATH.name)
        if ROC_PLOT_PATH.exists():
            st.image(str(ROC_PLOT_PATH), caption=ROC_PLOT_PATH.name)


if __name__ == "__main__":
    main()
