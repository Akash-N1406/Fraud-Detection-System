"""
Phase 8 — real-time ML pipeline (Kafka consumer + live scoring).
Run from src/: python kafka/consumer/transaction_consumer.py

Replaces Phase 7's placeholder process_transaction() with the actual
trained Random Forest model from Phase 6: computes the same engineered
features live, scores each transaction, and classifies risk per the SRS's
scale. Prediction latency is measured against NFR-02's <2 second target.
"""
import json
import time
from pathlib import Path

import joblib
import pandas as pd
from kafka import KafkaConsumer

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
MODEL_PATH = PROJECT_ROOT / "src" / "ml" / "models" / "random_forest_tuned.pkl"

TOPIC = "fraud-transactions"
BOOTSTRAP_SERVERS = "localhost:9092"
CONSUMER_GROUP = "fraud-detection-consumer"

# SRS section 7 risk classification thresholds
RISK_THRESHOLDS = [
    (0.30, "Low"),
    (0.60, "Medium"),
    (0.80, "High"),
    (1.01, "Critical"),  # 1.01 so a probability of exactly 1.0 still matches
]
ALERT_THRESHOLD = 0.75  # SRS section 10 example threshold


def classify_risk(probability: float) -> str:
    for upper_bound, label in RISK_THRESHOLDS:
        if probability <= upper_bound:
            return label
    return "Critical"  # unreachable given the 1.01 sentinel, but explicit


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"{MODEL_PATH} not found. Run Phase 6's "
            "ml/training/train_full_comparison.py first."
        )
    return joblib.load(MODEL_PATH)


def build_features(message: dict, expected_columns: list[str]) -> pd.DataFrame:
    """
    Compute the same engineered features as Phase 5/6, from one raw
    transaction message, then reindex to the model's exact expected
    column order — this reindex is what guarantees correctness regardless
    of whatever order training happened to produce; a plain dict-to-
    DataFrame conversion would silently risk misaligned columns.
    """
    error_balance_orig = message["newbalanceOrig"] + message["amount"] - message["oldbalanceOrg"]
    error_balance_dest = message["oldbalanceDest"] + message["amount"] - message["newbalanceDest"]

    row = {
        "amount": message["amount"],
        "oldbalanceOrg": message["oldbalanceOrg"],
        "newbalanceOrig": message["newbalanceOrig"],
        "oldbalanceDest": message["oldbalanceDest"],
        "newbalanceDest": message["newbalanceDest"],
        "errorBalanceOrig": error_balance_orig,
        "errorBalanceDest": error_balance_dest,
        "hour_of_day": message["step"] % 24,
        "day": message["step"] // 24,
        "type_CASH_OUT": 1 if message["type"] == "CASH_OUT" else 0,
    }

    df = pd.DataFrame([row])
    missing = set(expected_columns) - set(df.columns)
    if missing:
        raise ValueError(f"Feature engineering is missing columns the model expects: {missing}")

    return df[expected_columns]  # reindex to the model's exact training order


class RunningStats:
    """Tracks live accuracy against ground truth, for the demo/report only —
    a real deployment wouldn't have isFraud available at scoring time."""

    def __init__(self):
        self.total = 0
        self.actual_fraud = 0
        self.flagged = 0
        self.true_positives = 0
        self.false_positives = 0
        self.latencies_ms = []

    def update(self, predicted_fraud: bool, actual_fraud: bool, latency_ms: float):
        self.total += 1
        self.latencies_ms.append(latency_ms)
        if actual_fraud:
            self.actual_fraud += 1
        if predicted_fraud:
            self.flagged += 1
            if actual_fraud:
                self.true_positives += 1
            else:
                self.false_positives += 1

    def summary(self) -> str:
        avg_latency = sum(self.latencies_ms) / len(self.latencies_ms) if self.latencies_ms else 0
        max_latency = max(self.latencies_ms) if self.latencies_ms else 0
        recall = self.true_positives / self.actual_fraud if self.actual_fraud else 0.0
        precision = self.true_positives / self.flagged if self.flagged else 0.0
        return (
            f"[{self.total} processed] actual_fraud={self.actual_fraud} "
            f"flagged={self.flagged} recall={recall:.1%} precision={precision:.1%} "
            f"avg_latency={avg_latency:.1f}ms max_latency={max_latency:.1f}ms"
        )


def process_transaction(message: dict, model, expected_columns: list[str], stats: RunningStats) -> None:
    start = time.perf_counter()

    features = build_features(message, expected_columns)
    fraud_probability = model.predict_proba(features)[0, 1]
    risk_level = classify_risk(fraud_probability)
    is_flagged = fraud_probability > ALERT_THRESHOLD

    latency_ms = (time.perf_counter() - start) * 1000

    actual_fraud = bool(message.get("isFraud"))
    stats.update(predicted_fraud=is_flagged, actual_fraud=actual_fraud, latency_ms=latency_ms)

    alert_tag = " *** FRAUD ALERT ***" if is_flagged else ""
    ground_truth_tag = " [was actually fraud]" if actual_fraud else ""
    print(
        f"{message['transaction_id']} | {message['type']} | ${message['amount']:,.2f} | "
        f"P(fraud)={fraud_probability:.4f} | risk={risk_level} | "
        f"{latency_ms:.1f}ms{alert_tag}{ground_truth_tag}"
    )

    if stats.total % 25 == 0:
        print(f"  -- {stats.summary()}")


def main() -> None:
    print("Loading trained model...")
    model = load_model()
    expected_columns = list(model.feature_names_in_)
    print(f"Model loaded. Expected features: {expected_columns}")

    consumer = KafkaConsumer(
        TOPIC,
        bootstrap_servers=BOOTSTRAP_SERVERS,
        group_id=CONSUMER_GROUP,
        value_deserializer=lambda v: json.loads(v.decode("utf-8")),
        key_deserializer=lambda k: k.decode("utf-8") if k else None,
        auto_offset_reset="latest",
    )
    print(f"Listening on '{TOPIC}' at {BOOTSTRAP_SERVERS} (group: {CONSUMER_GROUP})...")
    print("Press Ctrl+C to stop.\n")

    stats = RunningStats()
    try:
        for record in consumer:
            process_transaction(record.value, model, expected_columns, stats)
    except KeyboardInterrupt:
        print(f"\nStopped.\nFinal: {stats.summary()}")
    finally:
        consumer.close()


if __name__ == "__main__":
    main()