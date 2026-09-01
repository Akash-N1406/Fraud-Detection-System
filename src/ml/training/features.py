"""
Feature preparation shared across all training scripts (baseline now, full
model comparison in Phase 6). Keeping this in one place means every model
sees an identical feature set, so comparisons in Phase 6 are fair.
"""
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
PROCESSED_PATH = PROJECT_ROOT / "data" / "processed" / "transfer_cashout_with_errors.parquet"

# Columns dropped before modeling:
#  - nameOrig/nameDest: high-cardinality identifiers, not generalizable features
#  - step: superseded by derived hour_of_day/day (avoids the model just memorizing
#    simulation time rather than learning a transferable pattern)
DROP_COLUMNS = ["nameOrig", "nameDest", "step"]
TARGET_COLUMN = "isFraud"


def load_features(path: Path = PROCESSED_PATH) -> tuple[pd.DataFrame, pd.Series]:
    """Load the Phase 2 processed dataset and return (X, y) ready for modeling."""
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run src/phase2_eda.ipynb first to generate it."
        )

    df = pd.read_parquet(path)

    # isFlaggedFraud is the simulator's naive heuristic (Phase 2 showed it barely
    # works) — excluding it forces the model to learn real signal rather than
    # leaning on a rule we already proved is weak.
    drop_cols = DROP_COLUMNS + ["isFlaggedFraud"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # hour_of_day/day may not exist if loading straight from Phase 2's saved
    # parquet before those columns were added there — derive defensively.
    if "hour_of_day" not in df.columns and "step" in df.columns:
        df["hour_of_day"] = df["step"] % 24
        df["day"] = df["step"] // 24

    # One-hot encode transaction type (only TRANSFER/CASH_OUT present post-Phase-2 filter)
    df = pd.get_dummies(df, columns=["type"], drop_first=True)

    y = df[TARGET_COLUMN]
    X = df.drop(columns=[TARGET_COLUMN])

    return X, y


if __name__ == "__main__":
    X, y = load_features()
    print(f"Feature matrix: {X.shape}")
    print(f"Columns: {list(X.columns)}")
    print(f"Fraud rate: {y.mean():.4%}")