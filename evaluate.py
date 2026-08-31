"""Evaluate the trained Logistic Regression model on the held-out test split.

Produces every metric the spec requires plus two plots:
    accuracy, confusion matrix, precision, recall, F1 (classification_report),
    ROC-AUC (from predict_proba)
    -> metrics.json, confusion_matrix.png, roc_curve.png

Positive class is 1 = "diabetes" (scikit-learn's default `pos_label=1`), which is
also the clinically meaningful direction: recall here is the share of genuine
diabetics the model catches. Per-class figures for both classes are stored too.

Run:  python evaluate.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")  # headless-safe: works in scripts and under Streamlit

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from data_utils import TARGET_NAMES

ROOT = Path(__file__).parent
METRICS_PATH = ROOT / "metrics.json"
CM_PLOT_PATH = ROOT / "confusion_matrix.png"
ROC_PLOT_PATH = ROOT / "roc_curve.png"

POSITIVE_CLASS = 1  # diabetes

# Sanity bands from the project spec, for the real Pima dataset.
ACC_BAND = (0.72, 0.80)
AUC_BAND = (0.78, 0.85)

CV_FOLDS = 5


def compute_metrics(model, X_test_scaled: np.ndarray, y_test, extra: dict | None = None) -> dict:
    """Every required metric, computed from a fitted model and the scaled test split."""
    y_pred = model.predict(X_test_scaled)
    pos_col = list(model.classes_).index(POSITIVE_CLASS)
    y_proba = model.predict_proba(X_test_scaled)[:, pos_col]

    labels = [0, 1]
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    fpr, tpr, thresholds = roc_curve(y_test, y_proba, pos_label=POSITIVE_CLASS)

    per_class = {
        TARGET_NAMES[c]: {
            "precision": float(precision_score(y_test, y_pred, pos_label=c, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, pos_label=c, zero_division=0)),
            "f1_score": float(f1_score(y_test, y_pred, pos_label=c, zero_division=0)),
            "support": int((np.asarray(y_test) == c).sum()),
        }
        for c in labels
    }

    tn, fp, fn, tp = (int(v) for v in cm.ravel())

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, pos_label=POSITIVE_CLASS, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, pos_label=POSITIVE_CLASS, zero_division=0)),
        "f1_score": float(f1_score(y_test, y_pred, pos_label=POSITIVE_CLASS, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_labels": list(TARGET_NAMES),
        "confusion_matrix_terms": {
            "true_negative": tn, "false_positive": fp,
            "false_negative": fn, "true_positive": tp,
        },
        "positive_class": TARGET_NAMES[POSITIVE_CLASS],
        "per_class": per_class,
        "classification_report": classification_report(
            y_test, y_pred, labels=labels, target_names=TARGET_NAMES, digits=4
        ),
        "classification_report_dict": classification_report(
            y_test, y_pred, labels=labels, target_names=TARGET_NAMES, output_dict=True
        ),
        "roc_curve": {"fpr": fpr.tolist(), "tpr": tpr.tolist()},
        "n_test": int(len(y_test)),
        "test_class_balance": {
            TARGET_NAMES[c]: int((np.asarray(y_test) == c).sum()) for c in labels
        },
        "sanity_bands": {"accuracy": list(ACC_BAND), "roc_auc": list(AUC_BAND)},
    }

    lo, hi = accuracy_ci(metrics["accuracy"], metrics["n_test"])
    metrics["accuracy_ci95"] = [lo, hi]
    metrics["sanity_check"] = {
        "accuracy_in_band": bool(ACC_BAND[0] <= metrics["accuracy"] <= ACC_BAND[1]),
        "accuracy_ci_overlaps_band": bool(hi >= ACC_BAND[0] and lo <= ACC_BAND[1]),
        "roc_auc_in_band": bool(AUC_BAND[0] <= metrics["roc_auc"] <= AUC_BAND[1]),
    }
    if extra:
        metrics.update(extra)
    return metrics


def accuracy_ci(accuracy: float, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% normal-approximation confidence interval for a test-set accuracy.

    With n=154 the standard error is ~0.037, so a single-split accuracy is a
    fairly noisy point estimate. Reported so the M6 sanity check can distinguish
    "the model is broken" from "this fold is unlucky".
    """
    se = float(np.sqrt(accuracy * (1.0 - accuracy) / n))
    return (max(0.0, accuracy - z * se), min(1.0, accuracy + z * se))


def cross_validated_accuracy(dataset=None, folds: int = CV_FOLDS) -> dict:
    """Stratified k-fold accuracy of the *identical* pipeline, on all 768 rows.

    The imputer and scaler are wrapped in a `Pipeline`, so scikit-learn refits
    them inside every fold — the CV number is therefore leakage-free too. This is
    the low-variance estimate used to sanity-check the model against the spec's
    0.72-0.80 accuracy band.
    """
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression

    from data_utils import RANDOM_STATE, load_data
    from train import MAX_ITER

    dataset = dataset or load_data()
    pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=MAX_ITER, random_state=RANDOM_STATE)),
        ]
    )
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(pipe, dataset.X, dataset.y, cv=cv, scoring="accuracy")
    auc_scores = cross_val_score(pipe, dataset.X, dataset.y, cv=cv, scoring="roc_auc")
    return {
        "folds": int(folds),
        "accuracy_scores": [float(s) for s in scores],
        "accuracy_mean": float(scores.mean()),
        "accuracy_std": float(scores.std()),
        "roc_auc_mean": float(auc_scores.mean()),
        "roc_auc_std": float(auc_scores.std()),
    }


def plot_confusion_matrix(cm, target_names: Sequence[str] = TARGET_NAMES):
    """Heatmap of the confusion matrix. Returns the matplotlib Figure."""
    cm = np.asarray(cm)
    fig, ax = plt.subplots(figsize=(5.0, 4.2))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", cbar=False, square=True,
        xticklabels=list(target_names), yticklabels=list(target_names), ax=ax,
    )
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Confusion Matrix (test set)")
    fig.tight_layout()
    return fig


def plot_roc_curve(fpr: Sequence[float], tpr: Sequence[float], roc_auc: float):
    """ROC curve with the chance diagonal. Returns the matplotlib Figure."""
    fig, ax = plt.subplots(figsize=(5.0, 4.2))
    ax.plot(fpr, tpr, color="#1f77b4", lw=2,
            label=f"Logistic Regression (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], color="grey", lw=1, ls="--", label="Chance (AUC = 0.5)")
    ax.set_xlim(-0.01, 1.0)
    ax.set_ylim(0.0, 1.01)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve (positive class: diabetes)")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def save_plots(metrics: dict) -> tuple[Path, Path]:
    fig = plot_confusion_matrix(metrics["confusion_matrix"])
    fig.savefig(CM_PLOT_PATH, dpi=150)
    plt.close(fig)

    fig = plot_roc_curve(metrics["roc_curve"]["fpr"], metrics["roc_curve"]["tpr"],
                         metrics["roc_auc"])
    fig.savefig(ROC_PLOT_PATH, dpi=150)
    plt.close(fig)
    return CM_PLOT_PATH, ROC_PLOT_PATH


def save_metrics(metrics: dict, path: Path = METRICS_PATH) -> Path:
    path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return path


def load_metrics(path: Path = METRICS_PATH) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def print_metrics(metrics: dict) -> None:
    print(f"=== M6: evaluation on held-out test set (n={metrics['n_test']}, "
          f"positive class = {metrics['positive_class']}) ===")
    print(f"Test class balance : {metrics['test_class_balance']}")
    print(f"Accuracy   : {metrics['accuracy']:.4f}")
    print(f"Precision  : {metrics['precision']:.4f}")
    print(f"Recall     : {metrics['recall']:.4f}")
    print(f"F1-score   : {metrics['f1_score']:.4f}")
    print(f"ROC-AUC    : {metrics['roc_auc']:.4f}")

    t = metrics["confusion_matrix_terms"]
    print(f"\nConfusion matrix (rows = true, cols = predicted, order "
          f"{metrics['confusion_matrix_labels']}):")
    for row in metrics["confusion_matrix"]:
        print(f"  {row}")
    print(f"  TN={t['true_negative']}  FP={t['false_positive']}  "
          f"FN={t['false_negative']}  TP={t['true_positive']}")

    print("\nClassification report:")
    print(metrics["classification_report"])

    acc_lo, acc_hi = metrics["sanity_bands"]["accuracy"]
    auc_lo, auc_hi = metrics["sanity_bands"]["roc_auc"]
    ci_lo, ci_hi = metrics["accuracy_ci95"]
    sc = metrics["sanity_check"]

    print("--- Spec sanity checks ---")
    print(f"ROC-AUC in [{auc_lo}, {auc_hi}]                      : "
          f"{'PASS' if sc['roc_auc_in_band'] else 'FAIL'} ({metrics['roc_auc']:.4f})")
    print(f"Fixed-split accuracy in [{acc_lo}, {acc_hi}]          : "
          f"{'PASS' if sc['accuracy_in_band'] else 'BELOW BAND'} ({metrics['accuracy']:.4f})")
    print(f"  its 95% CI [{ci_lo:.4f}, {ci_hi:.4f}] overlaps band : "
          f"{'PASS' if sc['accuracy_ci_overlaps_band'] else 'FAIL'}  "
          f"(n={metrics['n_test']}, SE~{(ci_hi - ci_lo) / 3.92:.4f})")

    cv = metrics.get("cross_validation")
    if cv:
        print(f"{cv['folds']}-fold CV accuracy in [{acc_lo}, {acc_hi}]         : "
              f"{'PASS' if ACC_BAND[0] <= cv['accuracy_mean'] <= ACC_BAND[1] else 'FAIL'} "
              f"({cv['accuracy_mean']:.4f} +- {cv['accuracy_std']:.4f})")
        print(f"{cv['folds']}-fold CV ROC-AUC  in [{auc_lo}, {auc_hi}]         : "
              f"{'PASS' if AUC_BAND[0] <= cv['roc_auc_mean'] <= AUC_BAND[1] else 'FAIL'} "
              f"({cv['roc_auc_mean']:.4f} +- {cv['roc_auc_std']:.4f})")
        print(f"  per-fold accuracy: "
              f"{[round(s, 4) for s in cv['accuracy_scores']]}")


def evaluate_saved_model() -> dict:
    """Rebuild the exact train/test split, evaluate the saved model, write artifacts."""
    import joblib

    from train import IMPUTER_PATH, MODEL_PATH, SCALER_PATH, TRAIN_INFO_PATH
    from data_utils import load_data, split_data

    if not MODEL_PATH.exists():
        raise SystemExit("model.pkl not found. Run `python train.py` first.")

    dataset = load_data()
    _, X_test, _, y_test = split_data(dataset)
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    imputer = joblib.load(IMPUTER_PATH)

    X_test_scaled = scaler.transform(imputer.transform(X_test))

    extra = {"data_source": dataset.source, "is_synthetic": dataset.is_synthetic}
    if TRAIN_INFO_PATH.exists():
        info = json.loads(TRAIN_INFO_PATH.read_text(encoding="utf-8"))
        extra["trained_at"] = info.get("trained_at")
        extra["n_train"] = info.get("n_train")

    extra["cross_validation"] = cross_validated_accuracy(dataset)

    metrics = compute_metrics(model, X_test_scaled, y_test, extra=extra)
    save_metrics(metrics)
    save_plots(metrics)
    return metrics


if __name__ == "__main__":
    m = evaluate_saved_model()
    print_metrics(m)
    print(f"\nSaved -> {METRICS_PATH.name}, {CM_PLOT_PATH.name}, {ROC_PLOT_PATH.name}")
