"""
Shared model evaluation. On a ~0.1% fraud rate, plain accuracy is close to
meaningless (predicting "not fraud" for everything scores ~99.9%), so every
model gets judged on precision/recall/F1/ROC-AUC/PR-AUC together, plus a
confusion matrix broken out explicitly.
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_model(model, X_test, y_test, model_name: str) -> dict:
    """Run predictions and compute the full metric set for one model."""
    y_pred = model.predict(X_test)

    # predict_proba isn't available on every estimator (e.g. some anomaly
    # detectors used from Phase 6 onward) — fall back gracefully
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        y_score = model.decision_function(X_test)
    else:
        y_score = y_pred

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    return {
        "model": model_name,
        "accuracy": round(accuracy_score(y_test, y_pred), 6),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 6),
        "recall": round(recall_score(y_test, y_pred, zero_division=0), 6),
        "f1_score": round(f1_score(y_test, y_pred, zero_division=0), 6),
        "roc_auc": round(roc_auc_score(y_test, y_score), 6),
        "pr_auc": round(average_precision_score(y_test, y_score), 6),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    }


def print_report(result: dict) -> None:
    print(f"\n{'=' * 50}")
    print(f"Model: {result['model']}")
    print(f"{'=' * 50}")
    print(f"Accuracy:  {result['accuracy']:.4%}")
    print(f"Precision: {result['precision']:.4%}  (of flagged transactions, how many were actually fraud)")
    print(f"Recall:    {result['recall']:.4%}  (of actual fraud, how much was caught)")
    print(f"F1 Score:  {result['f1_score']:.4f}")
    print(f"ROC-AUC:   {result['roc_auc']:.4f}")
    print(f"PR-AUC:    {result['pr_auc']:.4f}  (more informative than ROC-AUC under this imbalance)")
    print(f"\nConfusion Matrix:")
    print(f"  True Negatives:  {result['true_negatives']:>8,}   False Positives: {result['false_positives']:>6,}")
    print(f"  False Negatives: {result['false_negatives']:>8,}   True Positives:  {result['true_positives']:>6,}")


def save_comparison_report(results: list[dict], output_path: Path) -> None:
    """Write a markdown comparison table — one row per model — for the report."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(results)[
        ["model", "accuracy", "precision", "recall", "f1_score", "roc_auc", "pr_auc"]
    ]

    lines = ["# Model Comparison — Phase 3 Baseline\n", df.to_markdown(index=False)]
    output_path.write_text("\n".join(lines))

    json_path = output_path.with_suffix(".json")
    json_path.write_text(json.dumps(results, indent=2))

    print(f"\nSaved comparison report: {output_path}")
    print(f"Saved raw metrics JSON: {json_path}")