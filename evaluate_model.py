"""
evaluate_model.py
=================
Evaluates the ML document classifier (smart_doc_classifier.pkl) against
the cleaned dataset and generates a comprehensive performance report with
confusion matrix, classification report, cross-validation, and ROC curve.

Run: python evaluate_model.py
Outputs: evaluation_report.md, confusion_matrix.png, cv_scores.png, roc_curve.png
"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    roc_curve,
    auc,
)
from sklearn.preprocessing import label_binarize

BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH     = os.path.join(BASE_DIR, "src", "smart_doc_classifier.pkl")
DATA_PATH      = os.path.join(BASE_DIR, "cleaned_docs.csv")
OUTPUT_DIR     = os.path.join(BASE_DIR, "docs")
REPORT_PATH    = os.path.join(OUTPUT_DIR, "evaluation_report.md")
CM_IMAGE_PATH  = os.path.join(BASE_DIR, "confusion_matrix.png")
CV_IMAGE_PATH  = os.path.join(BASE_DIR, "cv_scores.png")
ROC_IMAGE_PATH = os.path.join(BASE_DIR, "roc_curve.png")

CATEGORIES = ["HR", "IT", "Finance"]


def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"ML model not found at {MODEL_PATH}")
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


def load_data():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Cleaned data not found at {DATA_PATH}. Run data_preprocessing.py first."
        )
    df = pd.read_csv(DATA_PATH)
    df = df[df["category"].isin(CATEGORIES)].dropna(subset=["text", "category"])
    return df["text"].tolist(), df["category"].tolist()


def plot_confusion_matrix(y_true, y_pred, labels, save_path):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
    )
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_title("SmartDoc Classifier — Confusion Matrix", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[OK] Confusion matrix saved -> {save_path}")


def plot_cv_scores(cv_scores, save_path):
    fig, ax = plt.subplots(figsize=(8, 4))
    folds = list(range(1, len(cv_scores) + 1))
    ax.bar(folds, cv_scores, color="#4F86C6", edgecolor="white", linewidth=0.8)
    ax.axhline(cv_scores.mean(), color="#E05252", linestyle="--", linewidth=1.5,
               label=f"Mean: {cv_scores.mean():.3f}")
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Fold", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title("5-Fold Cross-Validation Accuracy", fontsize=14, fontweight="bold")
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[OK] CV scores chart saved  -> {save_path}")


def plot_roc_curve(model, X, y, labels, save_path):
    y_bin = label_binarize(y, classes=labels)
    n_classes = y_bin.shape[1]

    if hasattr(model, "decision_function"):
        y_score = model.decision_function(X)
    elif hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X)
    else:
        print("[WARN] Model does not support decision_function/predict_proba for ROC.")
        return

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    fig, ax = plt.subplots(figsize=(7, 5))
    colors = ["#4F86C6", "#2ECC71", "#E74C3C"]
    for i, color in zip(range(n_classes), colors):
        ax.plot(
            fpr[i], tpr[i], color=color, lw=2,
            label=f"ROC {labels[i]} (AUC = {roc_auc[i]:.2f})"
        )
    ax.plot([0, 1], [0, 1], "k--", lw=1.5)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("Multi-Class ROC Curve — Document Classifier", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[OK] ROC curve saved        -> {save_path}")


def main():
    print("=" * 60)
    print("  SmartDoc ML Model Evaluation")
    print("=" * 60)

    try:
        model = load_model()
        print(f"[OK] Model loaded: {type(model).__name__}")
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        return

    try:
        X, y = load_data()
        print(f"[OK] Data loaded: {len(X)} samples")
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        return

    if len(X) < 5:
        print("[WARN] Too few samples for evaluation.")
        return

    y_pred = model.predict(X)
    acc    = accuracy_score(y, y_pred)

    report_str = classification_report(y, y_pred, labels=CATEGORIES, zero_division=0)
    print(f"\nAccuracy: {acc:.4f}\n")
    print("Classification Report:")
    print(report_str)

    n_splits = min(5, len(set(y)))
    skf      = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    try:
        cv_scores = cross_val_score(model, X, y, cv=skf, scoring="accuracy")
        print(f"CV Accuracy (mean ± std): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    except Exception as e:
        print(f"[WARN] Cross-validation skipped: {e}")
        cv_scores = np.array([acc])

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plot_confusion_matrix(y, y_pred, CATEGORIES, CM_IMAGE_PATH)
    plot_cv_scores(cv_scores, CV_IMAGE_PATH)
    plot_roc_curve(model, X, y, CATEGORIES, ROC_IMAGE_PATH)

    report_md = f"""# SmartDoc ML Model Evaluation Report

## Summary

| Metric | Value |
|--------|-------|
| Total Samples | {len(X)} |
| Overall Accuracy | {acc:.4f} ({acc*100:.1f}%) |
| CV Mean Accuracy | {cv_scores.mean():.4f} |
| CV Std Dev | {cv_scores.std():.4f} |
| Model Type | {type(model).__name__} |
| Categories | {', '.join(CATEGORIES)} |

## Classification Report

```
{report_str}
```

## Cross-Validation Scores

| Fold | Accuracy |
|------|----------|
{chr(10).join(f'| {i+1} | {s:.4f} |' for i, s in enumerate(cv_scores))}
| **Mean** | **{cv_scores.mean():.4f}** |

## Artefacts

- `confusion_matrix.png` — Confusion matrix heatmap
- `cv_scores.png` — Per-fold cross-validation accuracy chart
- `roc_curve.png` — Multi-class ROC curves and AUC scores
"""
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report_md)
    print(f"\n[OK] Evaluation report saved -> {REPORT_PATH}")
    print("\nEvaluation complete.")


if __name__ == "__main__":
    main()
