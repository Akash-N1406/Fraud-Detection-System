"""
Phase 5 — distributed feature engineering.
Run from src/: python spark/feature_engineering/build_features.py

Builds the same engineered features as the local pandas pipeline (Phases 2-3),
distributed across Spark. Keeping the feature *definitions* identical to
features.py is deliberate — Phase 6's model comparison should attribute any
performance difference to the algorithm, not to inconsistent features between
the local and distributed paths.
"""
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, floor, pmod

HDFS_PROCESSED_PATH = "hdfs://localhost:9000/fraud-detection/processed/paysim_cleaned.parquet"
HDFS_FEATURES_PATH = "hdfs://localhost:9000/fraud-detection/features/paysim_features.parquet"


def main() -> None:
    spark = (
        SparkSession.builder
        .appName("Phase5-FeatureEngineering")
        .master("local[*]")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    print(f"Reading cleaned data: {HDFS_PROCESSED_PATH}")
    df = spark.read.parquet(HDFS_PROCESSED_PATH)
    print(f"Row count: {df.count():,}")

    # --- Balance-consistency features -------------------------------
    # Same definition as the local pipeline: for a legitimate transaction,
    # newbalanceOrig should equal oldbalanceOrg - amount. Deviation from that
    # is the strongest fraud signal found in Phase 2/3 (43% RF importance).
    df = df.withColumn(
        "errorBalanceOrig",
        col("newbalanceOrig") + col("amount") - col("oldbalanceOrg"),
    )
    df = df.withColumn(
        "errorBalanceDest",
        col("oldbalanceDest") + col("amount") - col("newbalanceDest"),
    )

    # --- Time-based features -----------------------------------------
    # step = 1 simulated hour. pmod (not %) handles this safely for any
    # future negative-step edge case, though PaySim's step is always >= 0.
    df = df.withColumn("hour_of_day", pmod(col("step"), 24))
    df = df.withColumn("day", floor(col("step") / 24))

    # --- Type encoding --------------------------------------------------
    # Post-Phase-2 filtering, only TRANSFER/CASH_OUT remain, so a single
    # binary column captures the same information one-hot encoding would —
    # matches features.py's pd.get_dummies(..., drop_first=True) behavior
    df = df.withColumn(
        "type_CASH_OUT", when(col("type") == "CASH_OUT", 1).otherwise(0)
    )

    # --- Drop columns not used for modeling ------------------------------
    # Same exclusions as the local features.py: high-cardinality identifiers,
    # the raw `type`/`step` (superseded by derived columns), and the
    # already-debunked isFlaggedFraud heuristic from Phase 2
    df = df.drop("nameOrig", "nameDest", "step", "type", "isFlaggedFraud")

    print(f"Final feature columns: {df.columns}")

    print(f"Writing features: {HDFS_FEATURES_PATH}")
    # Same reasoning as clean_data.py — cap concurrent HDFS writers to avoid
    # overwhelming the single-node DataNode under local[*] parallelism
    df.coalesce(4).write.mode("overwrite").parquet(HDFS_FEATURES_PATH)

    print("Phase 5 feature engineering complete.")
    spark.stop()


if __name__ == "__main__":
    main()