"""
Phase 5 connectivity check.
Run from src/: python spark/verify_spark_hdfs.py

Confirms PySpark can start a local SparkSession and read a file from HDFS,
before any real preprocessing logic depends on that connection working.
"""
from pyspark.sql import SparkSession

HDFS_RAW_PATH = "hdfs://localhost:9000/fraud-detection/raw/PS_20174392719_1491204439457_log.csv"


def main() -> None:
    spark = (
        SparkSession.builder
        .appName("Phase5-ConnectivityCheck")
        .master("local[*]")  # local mode: no YARN, matches the Phase 4 scoping decision
        .getOrCreate()
    )
    # Reduce log noise — Spark's default INFO logging is very verbose for a sanity check
    spark.sparkContext.setLogLevel("WARN")

    print(f"Spark version: {spark.version}")
    print(f"Reading: {HDFS_RAW_PATH}")

    df = spark.read.csv(HDFS_RAW_PATH, header=True, inferSchema=True)

    print(f"Row count: {df.count():,}")
    print(f"Columns: {df.columns}")
    df.show(5, truncate=False)

    print("Phase 5 connectivity verified: PySpark can read from HDFS.")
    spark.stop()


if __name__ == "__main__":
    main()