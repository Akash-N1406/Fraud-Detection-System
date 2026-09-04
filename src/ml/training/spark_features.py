"""
Loads the Phase 5 Spark feature-engineering output (pulled from HDFS to
data/features/ via `hdfs dfs -get`) rather than recomputing features locally.
This is the canonical feature source for Phase 6 onward — using Spark's
actual output (not a local re-derivation) is what makes this a genuine
big-data pipeline rather than two parallel, coincidentally-similar paths.
"""
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
FEATURES_PATH = PROJECT_ROOT / "data" / "features" / "paysim_features.parquet"

TARGET_COLUMN = "isFraud"


def load_spark_features(path: Path = FEATURES_PATH) -> tuple[pd.DataFrame, pd.Series]:
    """Load Phase 5's Spark-engineered features and return (X, y)."""
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Pull it from HDFS first:\n"
            f"  mkdir -p data/features\n"
            f"  hdfs dfs -get /fraud-detection/features/paysim_features.parquet data/features/"
        )

    # path is a directory of Spark part-files — pandas reads this natively
    df = pd.read_parquet(path)

    y = df[TARGET_COLUMN]
    X = df.drop(columns=[TARGET_COLUMN])

    return X, y


if __name__ == "__main__":
    X, y = load_spark_features()
    print(f"Feature matrix: {X.shape}")
    print(f"Columns: {list(X.columns)}")
    print(f"Fraud rate: {y.mean():.4%}")