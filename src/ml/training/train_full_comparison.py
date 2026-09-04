"""
Phase 6 — full model development and comparison.
Run from src/: python ml/training/train_full_comparison.py

Extends Phase 3's baseline (Logistic Regression, Random Forest) with:
  - XGBoost, tuned via RandomizedSearchCV
  - Isolation Forest, an unsupervised anomaly detector (trained without
    labels, evaluated against them) — included per the SRS because it
    catches fraud patterns supervised models can't: novel fraud types
    not present in training data, since it doesn't need labeled examples
    of them to flag "this transaction doesn't look like the others."
  - Random Forest re-tuned via RandomizedSearchCV (Phase 3 used fixed
    hyperparameters; this searches around them)

Note: this trains on Phase 5's Spark-engineered features (data/features/),
not Phase 2's local pandas pipeline — see spark_features.py.
"""
import sys
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from ml.training.spark_features import load_spark_features
from ml.evaluation.metrics import evaluate_model, print_report, save_comparison_report

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
MODELS_DIR = PROJECT_ROOT / "src" / "ml" / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
RANDOM_STATE = 42

# RandomizedSearchCV uses average_precision (PR-AUC) as the scoring metric
# throughout — chosen in Phase 3 as the metric that actually distinguishes
# models under this ~0.1-0.3% fraud rate, unlike accuracy or plain ROC-AUC
SEARCH_SCORING = "average_precision"
SEARCH_CV = 3
SEARCH_N_ITER = 6  # kept small — full CV search over 2M+ rows is expensive;
                    # this explores enough of the space to beat fixed defaults
                    # without turning into an hours-long run

# Hyperparameter search runs on a stratified subsample, not the full 2.2M-row
# training set. Two reasons: (1) nesting n_jobs=-1 in both RandomizedSearchCV
# and the estimator itself causes process oversubscription — every core tries
# to spawn another full set of worker processes underneath it, which silently
# OOM-kills the process on memory-constrained environments like WSL2's default
# ~7-8GB allocation; (2) even without that crash, CV-searching the full
# dataset is unnecessarily slow. 300K rows is enough to find good
# hyperparameters; the winning config is then refit once on the full training
# set (a single fit, so full n_jobs=-1 parallelism there is safe).
SEARCH_SAMPLE_SIZE = 300_000


def stratified_subsample(X, y, n, random_state=RANDOM_STATE):
    """Stratified subsample for hyperparameter search (not for final fit)."""
    if len(X) <= n:
        return X, y
    X_sub, _, y_sub, _ = train_test_split(
        X, y, train_size=n, stratify=y, random_state=random_state
    )
    return X_sub, y_sub


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading Spark-engineered features...")
    X, y = load_spark_features()
    print(f"Feature matrix: {X.shape}  |  Fraud rate: {y.mean():.4%}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    print(f"Train: {X_train.shape}  |  Test: {X_test.shape}")

    results = []

    # ---------------------------------------------------------------
    # Random Forest — tuned on a subsample, refit on full training data
    # ---------------------------------------------------------------
    print("\n[1/4] Tuning Random Forest (on subsample)...")
    start = time.time()
    X_search, y_search = stratified_subsample(X_train, y_train, SEARCH_SAMPLE_SIZE)
    print(f"Search subsample: {X_search.shape}")

    rf_param_dist = {
        "n_estimators": [100, 200, 300],
        "max_depth": [8, 12, 16, 20],  # no None — unbounded trees on
                                          # millions of rows risk excessive
                                          # memory use
        "min_samples_leaf": [1, 2, 4],
    }
    # n_jobs=-1 on the search only, n_jobs=1 on the estimator — nesting both
    # causes process oversubscription (see SEARCH_SAMPLE_SIZE comment above)
    rf_search = RandomizedSearchCV(
        RandomForestClassifier(class_weight="balanced", random_state=RANDOM_STATE, n_jobs=1),
        rf_param_dist,
        n_iter=SEARCH_N_ITER,
        scoring=SEARCH_SCORING,
        cv=SEARCH_CV,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    rf_search.fit(X_search, y_search)
    print(f"Search done in {time.time() - start:.1f}s  |  Best params: {rf_search.best_params_}")

    print("Refitting best Random Forest config on full training set...")
    start = time.time()
    rf_best = RandomForestClassifier(
        class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1, **rf_search.best_params_
    )
    rf_best.fit(X_train, y_train)
    print(f"Refit in {time.time() - start:.1f}s")

    rf_result = evaluate_model(rf_best, X_test, y_test, "Random Forest (tuned)")
    print_report(rf_result)
    results.append(rf_result)
    joblib.dump(rf_best, MODELS_DIR / "random_forest_tuned.pkl")

    # ---------------------------------------------------------------
    # XGBoost — tuned on the same subsample, refit on full training data
    # ---------------------------------------------------------------
    print("\n[2/4] Tuning XGBoost (on subsample)...")
    start = time.time()
    # scale_pos_weight uses the FULL training set's imbalance ratio (not the
    # subsample's) since that's what the final full-data model actually sees
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    xgb_param_dist = {
        "n_estimators": [100, 200, 300],
        "max_depth": [4, 6, 8],
        "learning_rate": [0.05, 0.1, 0.2],
        "subsample": [0.8, 1.0],
    }
    xgb_search = RandomizedSearchCV(
        XGBClassifier(
            scale_pos_weight=scale_pos_weight,
            tree_method="hist",
            eval_metric="aucpr",
            random_state=RANDOM_STATE,
            n_jobs=1,  # non-nested — see rf_search comment
        ),
        xgb_param_dist,
        n_iter=SEARCH_N_ITER,
        scoring=SEARCH_SCORING,
        cv=SEARCH_CV,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    xgb_search.fit(X_search, y_search)
    print(f"Search done in {time.time() - start:.1f}s  |  Best params: {xgb_search.best_params_}")

    print("Refitting best XGBoost config on full training set...")
    start = time.time()
    xgb_best = XGBClassifier(
        scale_pos_weight=scale_pos_weight,
        tree_method="hist",
        eval_metric="aucpr",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        **xgb_search.best_params_,
    )
    xgb_best.fit(X_train, y_train)
    print(f"Refit in {time.time() - start:.1f}s")

    xgb_result = evaluate_model(xgb_best, X_test, y_test, "XGBoost (tuned)")
    print_report(xgb_result)
    results.append(xgb_result)
    joblib.dump(xgb_best, MODELS_DIR / "xgboost_tuned.pkl")

    # ---------------------------------------------------------------
    # Isolation Forest — unsupervised anomaly detection
    # ---------------------------------------------------------------
    print("\n[3/4] Training Isolation Forest...")
    start = time.time()
    # contamination = expected fraction of anomalies; using the actual
    # training fraud rate gives the model a realistic prior, even though
    # it never sees the labels during fitting
    contamination = y_train.mean()
    iso_forest = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    # Deliberately unsupervised: .fit(X_train) only, no y_train — this is
    # the entire point of including it. It has to find fraud purely from
    # "this transaction's feature pattern is unusual," not from labeled
    # examples, which is what makes it useful against fraud patterns a
    # supervised model was never trained on.
    iso_forest.fit(X_train)
    print(f"Trained in {time.time() - start:.1f}s")

    iso_result = evaluate_model(iso_forest, X_test, y_test, "Isolation Forest")
    print_report(iso_result)
    results.append(iso_result)
    joblib.dump(iso_forest, MODELS_DIR / "isolation_forest.pkl")

    # ---------------------------------------------------------------
    # Logistic Regression — re-run on Spark features for a fair comparison
    # ---------------------------------------------------------------
    print("\n[4/4] Re-running Logistic Regression on Spark features...")
    start = time.time()
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    log_reg = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE)
    log_reg.fit(X_train_scaled, y_train)
    print(f"Trained in {time.time() - start:.1f}s")

    lr_result = evaluate_model(log_reg, X_test_scaled, y_test, "Logistic Regression")
    print_report(lr_result)
    results.append(lr_result)
    joblib.dump(log_reg, MODELS_DIR / "logistic_regression_spark_features.pkl")

    # ---------------------------------------------------------------
    # Final comparison
    # ---------------------------------------------------------------
    save_comparison_report(
        results,
        REPORTS_DIR / "phase6_full_comparison.md",
        title="Model Comparison — Phase 6 Full Development",
    )

    print("\nAll models trained and compared. See reports/phase6_full_comparison.md")


if __name__ == "__main__":
    main()