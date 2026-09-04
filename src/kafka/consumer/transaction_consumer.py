"""
Phase 7 — transaction consumer skeleton (Kafka consumer).
Run from src/: python kafka/consumer/transaction_consumer.py

Deliberately minimal for this phase: it proves the producer -> Kafka ->
consumer pipeline works end to end, without any ML scoring yet. Phase 8
replaces the placeholder in process_transaction() with the actual trained
model, feature engineering, and risk classification.
"""
import json

from kafka import KafkaConsumer

TOPIC = "fraud-transactions"
BOOTSTRAP_SERVERS = "localhost:9092"
CONSUMER_GROUP = "fraud-detection-consumer"


def process_transaction(message: dict) -> None:
    """
    Placeholder — Phase 8 replaces this with:
      1. feature engineering (errorBalanceOrig/Dest, hour_of_day, day,
         type_CASH_OUT) matching Phase 5/6's feature definitions exactly
      2. model.predict_proba() using the Phase 6 Random Forest model
      3. risk classification (Low/Medium/High/Critical per the SRS scale)
      4. writing the prediction to PostgreSQL (Phase 9)

    For now, this just confirms the message arrived intact and reports
    whether it was actually fraud (ground truth, for pipeline sanity
    checking only — a real scorer wouldn't have this at inference time).
    """
    fraud_tag = " [ACTUAL FRAUD]" if message.get("isFraud") else ""
    print(
        f"Received {message['transaction_id']} | {message['type']} | "
        f"${message['amount']:,.2f} | step={message['step']}{fraud_tag}"
    )


def main() -> None:
    consumer = KafkaConsumer(
        TOPIC,
        bootstrap_servers=BOOTSTRAP_SERVERS,
        group_id=CONSUMER_GROUP,
        value_deserializer=lambda v: json.loads(v.decode("utf-8")),
        key_deserializer=lambda k: k.decode("utf-8") if k else None,
        auto_offset_reset="latest",  # only new messages — start the producer
                                       # first, or use "earliest" to replay
                                       # everything already in the topic
    )
    print(f"Listening on '{TOPIC}' at {BOOTSTRAP_SERVERS} (group: {CONSUMER_GROUP})...")
    print("Press Ctrl+C to stop.\n")

    count = 0
    try:
        for record in consumer:
            process_transaction(record.value)
            count += 1
    except KeyboardInterrupt:
        print(f"\nStopped. {count} messages processed.")
    finally:
        consumer.close()


if __name__ == "__main__":
    main()