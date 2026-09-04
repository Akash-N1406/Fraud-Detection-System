"""
Phase 7 — transaction simulator (Kafka producer).
Run from src/: python kafka/producer/transaction_simulator.py [--rate 2] [--limit 500]

Streams transactions sampled from the real PaySim dataset (TRANSFER/CASH_OUT
only, matching every earlier phase's scope) to the `fraud-transactions`
Kafka topic — one message per transaction, paced to simulate real-time
arrival rather than dumping everything at once.

`isFraud` is included in each message as ground truth for Phase 8's live
accuracy tracking (comparing the model's real-time prediction against what
actually happened) — a real production system wouldn't have this label
available at scoring time, but including it here lets the demo show
detection accuracy as transactions stream through, which is worth keeping
even though it's non-representative of a real deployment.
"""
import argparse
import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from kafka import KafkaProducer

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "PS_20174392719_1491204439457_log.csv"
TOPIC = "fraud-transactions"
BOOTSTRAP_SERVERS = "localhost:9092"
FRAUD_CAPABLE_TYPES = ["TRANSFER", "CASH_OUT"]


def load_sample_pool() -> pd.DataFrame:
    """Load and shuffle the TRANSFER/CASH_OUT subset once at startup."""
    print(f"Loading transaction pool from {RAW_DATA_PATH}...")
    df = pd.read_csv(RAW_DATA_PATH)
    df = df[df["type"].isin(FRAUD_CAPABLE_TYPES)]
    df = df.sample(frac=1, random_state=None).reset_index(drop=True)
    print(f"Pool ready: {len(df):,} transactions ({df['isFraud'].mean():.4%} fraud rate)")
    return df


def build_message(row: pd.Series) -> dict:
    """Convert one PaySim row into the streaming message schema."""
    return {
        "transaction_id": f"TXN{uuid.uuid4().hex[:10].upper()}",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "step": int(row["step"]),
        "type": row["type"],
        "amount": float(row["amount"]),
        "nameOrig": row["nameOrig"],
        "oldbalanceOrg": float(row["oldbalanceOrg"]),
        "newbalanceOrig": float(row["newbalanceOrig"]),
        "nameDest": row["nameDest"],
        "oldbalanceDest": float(row["oldbalanceDest"]),
        "newbalanceDest": float(row["newbalanceDest"]),
        "isFraud": int(row["isFraud"]),  # ground truth — see module docstring
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Stream PaySim transactions to Kafka")
    parser.add_argument("--rate", type=float, default=2.0, help="transactions per second")
    parser.add_argument("--limit", type=int, default=0, help="stop after N messages (0 = run until pool exhausted, then loop)")
    args = parser.parse_args()

    pool = load_sample_pool()
    delay = 1.0 / args.rate

    producer = KafkaProducer(
        bootstrap_servers=BOOTSTRAP_SERVERS,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        key_serializer=lambda k: k.encode("utf-8") if k else None,
    )
    print(f"Connected to Kafka at {BOOTSTRAP_SERVERS}, streaming to '{TOPIC}' at {args.rate}/s")

    sent = 0
    try:
        while True:
            for _, row in pool.iterrows():
                message = build_message(row)
                # keyed by nameOrig so all transactions from the same
                # origin account land on the same partition, preserving
                # per-account ordering — matters if Phase 8 ever tracks
                # per-account transaction velocity
                producer.send(TOPIC, key=message["nameOrig"], value=message)

                fraud_tag = " [FRAUD]" if message["isFraud"] else ""
                print(f"Sent {message['transaction_id']} | {message['type']} | ${message['amount']:,.2f}{fraud_tag}")

                sent += 1
                if args.limit and sent >= args.limit:
                    producer.flush()
                    print(f"\nLimit reached: {sent} messages sent.")
                    return

                time.sleep(delay)

            print("\nPool exhausted — reshuffling and continuing...")
            pool = pool.sample(frac=1).reset_index(drop=True)

    except KeyboardInterrupt:
        print(f"\nStopped by user. {sent} messages sent.")
    finally:
        producer.flush()
        producer.close()


if __name__ == "__main__":
    main()