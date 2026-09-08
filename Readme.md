# Distributed Big Data Analytics & Real-Time Fraud Detection System

A complete distributed platform for detecting fraudulent financial transactions, combining batch processing over historical data (Hadoop + Spark) with real-time streaming detection (Kafka + a trained ML model), backed by PostgreSQL, exposed through a Django REST API, and visualized in a live dashboard. Fully containerized deployment for the application tier via Docker Compose.

**Status: all 12 phases complete.**

## Overview

Financial institutions process millions of transactions daily, and fraud detection at that scale runs into real limits with centralized processing: slow batch jobs, poor horizontal scalability, and no way to react to a fraudulent transaction as it happens. This project addresses that with two complementary pipelines:

**Batch pipeline** — historical transactions land in HDFS, get cleaned and feature-engineered with PySpark, and are used to train and compare multiple fraud-classification models.

**Real-time pipeline** — simulated live transactions stream through Kafka, get scored against the trained model as they arrive, and generate a fraud probability and risk classification with average latency around 80ms (well under the 2-second target), persisted to PostgreSQL and surfaced on a live dashboard.

## Dataset

[PaySim](https://www.kaggle.com/datasets/ealaxi/paysim1) — a synthetic mobile-money transaction dataset (~6.3M rows) generated to resemble real financial logs without exposing actual customer data. Fraud is concentrated in `TRANSFER` and `CASH_OUT` transaction types, with severe class imbalance (~0.1–0.3% fraud rate) that shaped every metric and modeling decision throughout the project. See `src/phase2_eda.ipynb` for the full analysis.

## Tech Stack

| Component | Technology |
|---|---|
| Language | Python |
| Distributed storage | Hadoop HDFS (3.5.0, single-node, WSL2) |
| Batch processing | Apache Spark / PySpark (4.2.0) |
| Real-time streaming | Apache Kafka (4.3.1, KRaft mode — no ZooKeeper) |
| Machine learning | Scikit-learn, XGBoost |
| Backend / API | Django + Django REST Framework |
| Frontend | Django templates + Chart.js |
| Database | PostgreSQL 16 |
| Data processing | Pandas, NumPy |
| Containerization | Docker Compose (Postgres + Django tier) |
| Version control | Git/GitHub |

## Architecture

```text
                    DATA SOURCES
                         │
          ┌──────────────┴──────────────┐
          │                             │
          ▼                             ▼
    HISTORICAL DATA                LIVE DATA
          │                             │
          ▼                             ▼
        HDFS                        KAFKA
          │                             │
          ▼                             ▼
        SPARK                     CONSUMER
          │                             │
          └──────────────┬──────────────┘
                         │
                         ▼
                 FEATURE ENGINEERING
                         │
                         ▼
                    ML MODEL
              (Random Forest, tuned)
                         │
              ┌──────────┴─────────┐
              ▼                    ▼
        FRAUD PREDICTION       RISK SCORE
              │                    │
              └──────────┬─────────┘
                         ▼
                    POSTGRESQL
                         │
                         ▼
                DJANGO REST API
                         │
                         ▼
              DJANGO DASHBOARD (Chart.js)
```

## Project Structure

```text
fraud-detection-system/
├── data/
│   ├── raw/                       # original PaySim CSV (gitignored)
│   ├── processed/                 # Phase 2 cleaned data (gitignored)
│   └── features/                  # Phase 5 Spark features, pulled from HDFS (gitignored)
├── database/
│   └── schema.sql                 # Phase 9 PostgreSQL schema
├── src/
│   ├── hadoop/
│   │   └── verify_hdfs.py
│   ├── spark/
│   │   ├── preprocessing/clean_data.py
│   │   └── feature_engineering/build_features.py
│   ├── kafka/
│   │   ├── producer/transaction_simulator.py
│   │   └── consumer/transaction_consumer.py    # live ML scoring + DB writes
│   ├── ml/
│   │   ├── training/                # features.py, spark_features.py,
│   │   │                             #   train_baseline.py, train_full_comparison.py
│   │   ├── models/                  # saved model files (gitignored)
│   │   └── evaluation/metrics.py
│   ├── database/
│   │   ├── db.py                    # connection + insert helpers
│   │   └── load_model_metrics.py
│   ├── backend/                     # Django project
│   │   ├── backend/                 # settings, urls
│   │   ├── fraud_api/                # DRF models/serializers/views (REST API)
│   │   ├── dashboard/                 # template views + Chart.js frontend
│   │   ├── Dockerfile
│   │   └── requirements.txt
│   ├── verify_setup.py
│   └── phase2_eda.ipynb
├── docker-compose.yml
├── requirements.txt                  # data/ML pipeline dependencies
├── .env.example
└── README.md
```

## Before You Start Working (each WSL session)

HDFS and Kafka do **not** persist across WSL sessions — a closed terminal, WSL restart, or Windows sleep/reboot kills them. Anything touching them will fail until restarted:

```bash
# HDFS
start-dfs.sh
jps   # confirm NameNode, DataNode, SecondaryNameNode

# Kafka
cd $KAFKA_HOME && bin/kafka-server-start.sh config/server.properties

# PostgreSQL (if running natively rather than via Docker)
sudo service postgresql start
```

## Setup (WSL2 / Ubuntu)

```bash
# clone and enter
git clone https://github.com/Akash-N1406/Fraud-Detection-System.git fraud-detection-system
cd fraud-detection-system

# python environment
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# dataset — requires a Kaggle API token at ~/.kaggle/kaggle.json
kaggle datasets download -d ealaxi/paysim1 -p data/raw
cd data/raw && unzip paysim1.zip && rm paysim1.zip && cd ../..

# confirm environment + dataset
cd src && python verify_setup.py && cd ..
```

Hadoop, Spark, and Kafka setup are documented in full in the phase notes below — see Phases 4, 5, and 7.

## Running the Full Pipeline

**1. Batch pipeline** (Hadoop/Spark — Phases 4–6):
```bash
cd src
python hadoop/verify_hdfs.py
python spark/preprocessing/clean_data.py
python spark/feature_engineering/build_features.py
python ml/training/train_full_comparison.py
```

**2. Real-time pipeline** (Kafka — Phases 7–9), in separate terminals:
```bash
# terminal A
python kafka/consumer/transaction_consumer.py

# terminal B
python kafka/producer/transaction_simulator.py --rate 5 --limit 500
```

**3. API + dashboard** (Django — Phases 10–11):
```bash
cd src/backend
python manage.py migrate
python manage.py createsuperuser   # first time only
python manage.py runserver
```
Visit `http://localhost:8000/dashboard/` and log in.

**4. Or, containerized app tier** (Docker — Phase 12):
```bash
docker compose up --build
```
Starts Postgres (host port 5433) and Django (host port 8001) in containers — a fresh, empty database by default. See `docker-compose.yml`'s header comment for restoring real data into it. Hadoop/Kafka/Spark still run natively per steps 1–2 above; see the Architecture Decisions section below for why.

## Development Phases — All Complete

| Phase | Focus | Key Result |
|---|---|---|
| 1 | Project setup | Repo, venv, dataset, Docker Desktop confirmed |
| 2 | Exploratory data analysis | Fraud confined to TRANSFER/CASH_OUT; `isFlaggedFraud` heuristic catches almost none of it; balance-error features identified as strong signal |
| 3 | ML baseline | Random Forest F1=0.998 vs. Logistic Regression F1=0.09 on the same features |
| 4 | Hadoop/HDFS | Single-node HDFS (3.5.0) on WSL2, 470.7MB raw + 132MB processed data migrated |
| 5 | Spark processing | Distributed preprocessing/feature engineering, 2,770,409 rows, matches local pipeline exactly |
| 6 | Model development | Full comparison incl. XGBoost and Isolation Forest; tuned Random Forest remains best (F1=0.998); Isolation Forest badly underperforms (F1=0.026) — see Key Findings |
| 7 | Kafka streaming | KRaft-mode Kafka (4.3.1, no ZooKeeper), producer/consumer verified end-to-end |
| 8 | Real-time ML pipeline | Live scoring via the trained model, ~80ms average latency, feature-column reindexing verified correct |
| 9 | Database integration | PostgreSQL persistence; live 2000-transaction stream caught 4/4 real fraud cases with zero false alerts |
| 10 | Django backend | REST API over the live data, token auth, alert triage endpoint |
| 11 | Frontend dashboard | KPI cards, 4 charts, transaction table, alert triage UI — all Chart.js + Django templates |
| 12 | Docker deployment | Postgres + Django containerized via Docker Compose; confirmed working on first run |

## Key Findings

- Fraud occurs **only** in `TRANSFER` and `CASH_OUT` transaction types — the other three types contain zero fraud in this dataset.
- The dataset's built-in `isFlaggedFraud` heuristic (flag TRANSFER > 200,000) catches only a small fraction of actual fraud — the core motivation for the ML approach.
- Balance-consistency errors (`errorBalanceOrig`, `errorBalanceDest`) are the dominant predictive signal (43% of Random Forest's feature importance). This is partly a known artifact of how PaySim's simulator generates fraudulent transactions, not purely a real-world-generalizable pattern — worth stating plainly rather than overclaiming from a near-perfect test score.
- **Isolation Forest, an unsupervised anomaly detector, badly underperforms** the supervised models (F1=0.026 vs. 0.998) despite "anomaly detection" sounding well-suited to fraud. With many features that vary widely for legitimate transactions too, random partitioning dilutes the one dominant, narrow, learnable signal that supervised models can weight directly. A genuinely useful negative result: unsupervised methods don't automatically help just because the problem sounds like anomaly detection.
- Random Forest and XGBoost both achieve ~99.6% recall on true fraud; Random Forest's precision (100%, zero false positives) edges out XGBoost's (84.9%) for this specific problem.
- The full live pipeline (Kafka → real-time scoring → PostgreSQL) was verified against a genuinely new, independently streamed 2000-transaction batch — not just the held-out test split — and matched the offline test metrics almost exactly (4/4 real fraud caught, 0 false alerts).

## Architecture Decisions Worth Knowing

**Why HDFS/Kafka/Spark aren't in Docker Compose:** these services need to be reachable both from inside a Docker network and from host-side scripts (the producer/consumer, Spark batch jobs). Correctly configuring that dual-access networking (advertised listeners, hostname resolution) is a materially different and riskier problem than containerizing a stateless web tier, and wasn't necessary to get real reproducible-deployment value out of Phase 12. The stateful distributed infrastructure runs natively; the stateless application tier is containerized — a defensible split, not a shortcut.

**Why the ML pipeline reads Spark's actual output, not a re-derived local copy** (`ml/training/spark_features.py`): using the genuine Phase 5 pipeline output as the Phase 6 training source — rather than recomputing features locally — is what makes this a real end-to-end big-data pipeline instead of two parallel, coincidentally similar paths.

**Why the Django models are `managed=False`:** the PostgreSQL tables are the Kafka consumer's tables first — Django only reads and serves them via the API. Letting Django "manage" tables that already have live data flowing into them would risk migration conflicts.