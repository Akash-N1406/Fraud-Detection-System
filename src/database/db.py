"""
Phase 9 — PostgreSQL integration for the real-time pipeline.
Used by kafka/consumer/transaction_consumer.py to persist every
transaction, prediction, and (when flagged) fraud alert.

Connection settings come from environment variables, with sane local
defaults — set these in a .env file (see .env.example) rather than
hardcoding credentials in source.
"""
import os
from contextlib import contextmanager

import psycopg2
from dotenv import load_dotenv
from psycopg2.extras import RealDictCursor

# Loads .env from the project root if present — falls back to already-set
# shell environment variables (e.g. exported manually, or in CI) if not.
load_dotenv()

DB_CONFIG = {
    "host": os.environ.get("DB_HOST", "localhost"),
    "port": os.environ.get("DB_PORT", "5432"),
    "dbname": os.environ.get("DB_NAME", "frauddb"),
    "user": os.environ.get("DB_USER", "postgres"),
    "password": os.environ.get("DB_PASSWORD", "postgres"),
}


@contextmanager
def get_connection():
    """Context-managed connection — commits on success, rolls back on error."""
    conn = psycopg2.connect(**DB_CONFIG)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def insert_transaction(conn, message: dict) -> None:
    """
    Insert the raw transaction. ON CONFLICT DO NOTHING makes this safe to
    call even if the same transaction_id somehow arrives twice (e.g. a
    consumer restart re-reading uncommitted offsets) — the unique
    constraint on transaction_id would otherwise raise on a duplicate.
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO transactions (
                transaction_id, step, type, amount, name_orig,
                old_balance_org, new_balance_orig, name_dest,
                old_balance_dest, new_balance_dest, is_fraud_actual
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (transaction_id) DO NOTHING
            """,
            (
                message["transaction_id"],
                message["step"],
                message["type"],
                message["amount"],
                message["nameOrig"],
                message["oldbalanceOrg"],
                message["newbalanceOrig"],
                message["nameDest"],
                message["oldbalanceDest"],
                message["newbalanceDest"],
                bool(message.get("isFraud")),
            ),
        )


def insert_prediction(
    conn, transaction_id: str, fraud_probability: float, risk_level: str,
    model_version: str, latency_ms: float,
) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO predictions (
                transaction_id, fraud_probability, risk_level,
                model_version, latency_ms
            ) VALUES (%s, %s, %s, %s, %s)
            """,
            (transaction_id, fraud_probability, risk_level, model_version, latency_ms),
        )


def insert_alert(conn, transaction_id: str, alert_level: str, fraud_probability: float) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO fraud_alerts (transaction_id, alert_level, fraud_probability)
            VALUES (%s, %s, %s)
            """,
            (transaction_id, alert_level, fraud_probability),
        )


def insert_model_metrics(conn, metrics: dict) -> None:
    """Load one model's Phase 6 evaluation metrics into model_metrics —
    used by scripts/load_model_metrics.py to populate the dashboard."""
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO model_metrics (
                model_name, accuracy, precision_score, recall_score,
                f1_score, roc_auc, pr_auc
            ) VALUES (%s, %s, %s, %s, %s, %s, %s)
            """,
            (
                metrics["model"],
                metrics["accuracy"],
                metrics["precision"],
                metrics["recall"],
                metrics["f1_score"],
                metrics["roc_auc"],
                metrics["pr_auc"],
            ),
        )


if __name__ == "__main__":
    # Quick connectivity check: python database/db.py
    with get_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT count(*) AS n FROM transactions")
            print(f"Connected. transactions table has {cur.fetchone()['n']} rows.")