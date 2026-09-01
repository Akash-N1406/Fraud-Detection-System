# Distributed Big Data Analytics & Real-Time Fraud Detection System

A distributed platform for detecting fraudulent financial transactions, combining
batch processing over historical data with near-real-time streaming analysis.

Built as a 12-phase project spanning distributed storage, big data processing,
machine learning, event streaming, and a full-stack analytics dashboard.

## Overview

Financial institutions process millions of transactions daily, and fraud detection
at that scale runs into real limits with centralized processing: slow batch jobs,
poor horizontal scalability, and no way to react to a fraudulent transaction as it
happens. This project addresses that with two complementary pipelines:

**Batch pipeline** — historical transactions land in HDFS, get cleaned and
feature-engineered with PySpark, and are used to train and compare multiple
fraud-classification models.

**Real-time pipeline** — simulated live transactions stream through Kafka, get
scored against the trained model as they arrive, and generate a fraud probability
and risk classification with sub-2-second latency, surfaced on a Django dashboard.

## Dataset

[PaySim](https://www.kaggle.com/datasets/ealaxi/paysim1) — a synthetic mobile-money
transaction dataset (~6.3M rows) generated to resemble real financial logs without
exposing actual customer data. Fraud is concentrated in `TRANSFER` and `CASH_OUT`
transaction types, with severe class imbalance (~0.1% fraud rate) that shapes the
metric choices and modeling approach throughout the project. See
`src/phase2_eda.ipynb` for the full analysis behind these decisions.

## Tech Stack

| Component | Technology |
|---|---|
| Language | Python |
| Distributed storage | Hadoop HDFS |
| Batch processing | Apache Spark / PySpark |
| Real-time streaming | Apache Kafka |
| Machine learning | Scikit-learn, XGBoost |
| Backend / API | Django |
| Database | PostgreSQL |
| Data processing | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn, Chart.js |
| Containerization | Docker / Docker Compose |

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
                 DJANGO DASHBOARD
```

## Project Structure

```text
fraud-detection-system/
├── data/
│   ├── raw/                  # original PaySim CSV (gitignored)
│   └── processed/            # cleaned/feature-engineered outputs (gitignored)
├── src/
│   ├── hadoop/
│   ├── spark/
│   │   ├── preprocessing/
│   │   ├── feature_engineering/
│   │   └── analytics/
│   ├── kafka/
│   │   ├── producer/
│   │   └── consumer/
│   ├── ml/
│   │   ├── training/
│   │   ├── models/            # saved model files (gitignored)
│   │   └── evaluation/
│   ├── backend/                # Django project
│   ├── verify_setup.py
│   └── phase2_eda.ipynb
├── database/
├── docker/
├── reports/
├── docs/
├── requirements.txt
└── README.md
```

## Setup (WSL2 / Ubuntu)

```bash
# clone and enter
git clone <repo-url> fraud-detection-system
cd fraud-detection-system

# python environment (3.10/3.11 recommended for Spark/PySpark compatibility)
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# dataset — requires a Kaggle API token at ~/.kaggle/kaggle.json
pip install kaggle
kaggle datasets download -d ealaxi/paysim1 -p data/raw
cd data/raw && unzip paysim1.zip && rm paysim1.zip && cd ../..

# confirm environment + dataset
cd src && python verify_setup.py
```

Docker Desktop with WSL2 integration is required from Phase 12 onward (and
optionally earlier for Hadoop/Kafka/Postgres containers) — see
`docker --version` / `docker ps` to confirm it's available.

## Development Phases

| Phase | Focus | Status |
|---|---|---|
| 1 | Project setup — repo, venv, dataset, Docker | ✅ Done |
| 2 | Exploratory data analysis | ✅ Done |
| 3 | ML baseline (Logistic Regression, Random Forest) | Not started |
| 4 | Big data integration — Hadoop/HDFS | Not started |
| 5 | Spark processing — distributed preprocessing & feature engineering | Not started |
| 6 | Model development — full comparison incl. XGBoost, Isolation Forest | Not started |
| 7 | Kafka streaming — producer/consumer, transaction simulator | Not started |
| 8 | Real-time ML pipeline | Not started |
| 9 | Database integration — PostgreSQL | Not started |
| 10 | Django backend — REST APIs, auth | Not started |
| 11 | Frontend dashboard — analytics, alerts, charts | Not started |
| 12 | Docker deployment — Compose, multi-container orchestration | Not started |

## Key Findings So Far (Phase 2)

- Fraud occurs **only** in `TRANSFER` and `CASH_OUT` transaction types.
- The dataset's built-in `isFlaggedFraud` heuristic (flag TRANSFER > 200,000)
  catches only a small fraction of actual fraud — motivating the ML approach.
- Balance-consistency errors (`errorBalanceOrig`, `errorBalanceDest`) show
  visibly different distributions between fraud and legitimate transactions,
  and are carried forward as engineered features.

Full analysis: `src/phase2_eda.ipynb`.