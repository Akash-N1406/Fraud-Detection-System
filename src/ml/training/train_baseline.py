"""
Phase 3 — ML Baseline.
Run from src/: python ml/training/train_baseline.py

Trains Logistic Regression and Random Forest on the Phase 2 processed dataset,
handling the ~0.1% fraud class imbalance via class_weight="balanced" rather
than resampling (keeps this baseline simple; SMOTE/undersampling comparisons
are a natural Phase 6 extension once we're comparing more models anyway).
"""
import sys
import time
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Allow running this script directly (python ml/training/train_baseline.py from src/)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from ml.training.features import load_features
from ml.evaluation.metrics import evaluate_model, print_report, save_comparison_report

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
MODELS_DIR = PROJECT_ROOT / "src" / "ml" / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
RANDOM_STATE = 42


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading features...")
    X, y = load_features()
    print(f"Feature matrix: {X.shape}  |  Fraud rate: {y.mean():.4%}")

    # stratify=y is essential here — a plain random split risks a test set
    # with almost no fraud cases at all given the ~0.1% base rate
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    print(f"Train: {X_train.shape}  |  Test: {X_test.shape}")
    print(f"Train fraud count: {y_train.sum():,}  |  Test fraud count: {y_test.sum():,}")

    results = []

    # ---------------------------------------------------------------
    # Logistic Regression — fast, interpretable baseline
    # ---------------------------------------------------------------
    print("\nTraining Logistic Regression...")
    # Logistic Regression is scale-sensitive (unlike tree models), so it
    # needs standardized features — fit the scaler on train only to avoid
    # leaking test-set statistics into training
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    start = time.time()
    log_reg = LogisticRegression(
        class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE
    )
    log_reg.fit(X_train_scaled, y_train)
    print(f"Trained in {time.time() - start:.1f}s")

    lr_result = evaluate_model(log_reg, X_test_scaled, y_test, "Logistic Regression")
    print_report(lr_result)
    results.append(lr_result)

    joblib.dump(log_reg, MODELS_DIR / "logistic_regression.pkl")
    joblib.dump(scaler, MODELS_DIR / "scaler.pkl")  # needed at inference time too

    # ---------------------------------------------------------------
    # Random Forest — handles nonlinearity, no scaling needed
    # ---------------------------------------------------------------
    print("\nTraining Random Forest...")
    start = time.time()
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        class_weight="balanced",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    rf.fit(X_train, y_train)
    print(f"Trained in {time.time() - start:.1f}s")

    rf_result = evaluate_model(rf, X_test, y_test, "Random Forest")
    print_report(rf_result)
    results.append(rf_result)

    joblib.dump(rf, MODELS_DIR / "random_forest.pkl")

    # ---------------------------------------------------------------
    # Feature importance (Random Forest only — Logistic Regression's
    # coefficients aren't directly comparable post-scaling without more care)
    # ---------------------------------------------------------------
    importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
    print("\nRandom Forest — Top feature importances:")
    print(importances.head(10).to_string())

    # ---------------------------------------------------------------
    # Save comparison report
    # ---------------------------------------------------------------
    save_comparison_report(results, REPORTS_DIR / "phase3_baseline_comparison.md")


if __name__ == "__main__":
    main()