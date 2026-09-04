"""
Phase 5 — distributed preprocessing.
Run from src/: python spark/preprocessing/clean_data.py

Reads the raw PaySim CSV from HDFS, applies the same cleaning logic that
Phase 2's pandas EDA established, and writes a cleaned Parquet back to HDFS.
This replicates Phase 2's findings at distributed scale rather than
re-deriving them — the filtering decisions were already validated there.
"""
from pyspark.sql import SparkSession
from pyspark.sql.types import (
    StructType, StructField, IntegerType, StringType, DoubleType
)
from pyspark.sql.functions import col

HDFS_RAW_PATH = "hdfs://localhost:9000/fraud-detection/raw/PS_20174392719_1491204439457_log.csv"
HDFS_PROCESSED_PATH = "hdfs://localhost:9000/fraud-detection/processed/paysim_cleaned.parquet"

# Explicit schema instead of inferSchema=True: inferSchema forces Spark to do
# a full extra pass over the 6.3M-row file just to guess types, which is both
# slower and less reliable than declaring the known PaySim schema directly.
SCHEMA = StructType([
    StructField("step", IntegerType(), False),
    StructField("type", StringType(), False),
    StructField("amount", DoubleType(), False),
    StructField("nameOrig", StringType(), False),
    StructField("oldbalanceOrg", DoubleType(), False),
    StructField("newbalanceOrig", DoubleType(), False),
    StructField("nameDest", StringType(), False),
    StructField("oldbalanceDest", DoubleType(), False),
    StructField("newbalanceDest", DoubleType(), False),
    StructField("isFraud", IntegerType(), False),
    StructField("isFlaggedFraud", IntegerType(), False),
])

# Confirmed in Phase 2 EDA: fraud occurs only in these two transaction types.
# Filtering here (not just at feature-engineering time) means every downstream
# Spark job reads a ~8x smaller file, which matters once this scales further.
FRAUD_CAPABLE_TYPES = ["TRANSFER", "CASH_OUT"]


def main() -> None:
    spark = (
        SparkSession.builder
        .appName("Phase5-Preprocessing")
        .master("local[*]")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    print(f"Reading raw data: {HDFS_RAW_PATH}")
    df = spark.read.csv(HDFS_RAW_PATH, header=True, schema=SCHEMA)
    raw_count = df.count()
    print(f"Raw row count: {raw_count:,}")

    # --- Cleaning ---------------------------------------------------
    # Drop exact-duplicate rows (defensive — PaySim shouldn't have any, but
    # a real pipeline should never assume that silently)
    df = df.dropDuplicates()

    # Drop rows with nulls in any required column — a malformed row shouldn't
    # silently propagate into feature engineering or model training
    df = df.dropna(subset=[f.name for f in SCHEMA.fields])

    # Restrict to the transaction types where fraud actually occurs
    df = df.filter(col("type").isin(FRAUD_CAPABLE_TYPES))

    cleaned_count = df.count()
    fraud_count = df.filter(col("isFraud") == 1).count()
    print(f"Cleaned row count: {cleaned_count:,} ({cleaned_count / raw_count:.1%} of raw)")
    print(f"Fraud rows retained: {fraud_count:,}")

    # --- Write ---------------------------------------------------------
    # Partitioning by `type` speeds up any downstream query that filters on
    # it (e.g. "TRANSFER-only" analysis) without materializing separate files.
    #
    # coalesce(4): local[*] mode uses all CPU cores, which on a single-node
    # WSL2 DataNode means 14-16+ simultaneous HDFS block-write streams —
    # this can overwhelm the DataNode's connection handling and trigger
    # lease/addBlock failures. Capping concurrent writers to 4 keeps this
    # reliable without meaningfully hurting throughput on a dataset this size.
    print(f"Writing cleaned data: {HDFS_PROCESSED_PATH}")
    (
        df.coalesce(4)
        .write
        .mode("overwrite")
        .partitionBy("type")
        .parquet(HDFS_PROCESSED_PATH)
    )

    print("Phase 5 preprocessing complete.")
    spark.stop()


if __name__ == "__main__":
    main()