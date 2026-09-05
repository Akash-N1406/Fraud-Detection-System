-- Phase 9 — Database schema
-- Run once against a fresh database: psql -U postgres -d frauddb -f database/schema.sql
--
-- Adapted from the SRS's example schema (section 12) to match our actual
-- PaySim fields — the SRS's sample schema assumed columns (merchant_category,
-- location, device_id) that don't exist in PaySim; this schema uses the
-- fields we actually have (type, nameOrig/nameDest, step) instead of
-- inventing data that was never collected.

CREATE TABLE IF NOT EXISTS transactions (
    id BIGSERIAL PRIMARY KEY,
    transaction_id VARCHAR(32) UNIQUE NOT NULL,
    step INTEGER NOT NULL,
    type VARCHAR(20) NOT NULL,
    amount NUMERIC(18, 2) NOT NULL,
    name_orig VARCHAR(32) NOT NULL,
    old_balance_org NUMERIC(18, 2) NOT NULL,
    new_balance_orig NUMERIC(18, 2) NOT NULL,
    name_dest VARCHAR(32) NOT NULL,
    old_balance_dest NUMERIC(18, 2) NOT NULL,
    new_balance_dest NUMERIC(18, 2) NOT NULL,
    is_fraud_actual BOOLEAN, -- ground truth, from the simulator only —
    -- a real system would not have this at
    -- ingestion time
    received_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS predictions (
    id BIGSERIAL PRIMARY KEY,
    transaction_id VARCHAR(32) NOT NULL REFERENCES transactions (transaction_id),
    fraud_probability NUMERIC(6, 5) NOT NULL,
    risk_level VARCHAR(10) NOT NULL,
    model_version VARCHAR(50) NOT NULL,
    latency_ms NUMERIC(8, 2),
    prediction_time TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS fraud_alerts (
    id BIGSERIAL PRIMARY KEY,
    transaction_id VARCHAR(32) NOT NULL REFERENCES transactions (transaction_id),
    alert_level VARCHAR(10) NOT NULL,
    fraud_probability NUMERIC(6, 5) NOT NULL,
    status VARCHAR(20) NOT NULL DEFAULT 'New',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS model_metrics (
    id BIGSERIAL PRIMARY KEY,
    model_name VARCHAR(50) NOT NULL,
    accuracy NUMERIC(6, 5),
    precision_score NUMERIC(6, 5), -- "precision" is a reserved-ish
    -- word in some contexts; named
    -- explicitly to avoid ambiguity
    recall_score NUMERIC(6, 5),
    f1_score NUMERIC(6, 5),
    roc_auc NUMERIC(6, 5),
    pr_auc NUMERIC(6, 5),
    training_date TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Indexes for the dashboard queries Phase 11 will run: recent alerts,
-- lookups by transaction, and time-ordered scans
CREATE INDEX IF NOT EXISTS idx_predictions_transaction_id ON predictions (transaction_id);

CREATE INDEX IF NOT EXISTS idx_fraud_alerts_transaction_id ON fraud_alerts (transaction_id);

CREATE INDEX IF NOT EXISTS idx_fraud_alerts_status ON fraud_alerts (status);

CREATE INDEX IF NOT EXISTS idx_transactions_received_at ON transactions (received_at);