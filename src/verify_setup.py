"""
Phase 1 sanity check.
Run from src/: python verify_setup.py
Confirms: dataset present, loads correctly, matches expected PaySim schema.
"""
from pathlib import Path
import pandas as pd

# Recurring project pattern: script runs from src/, resolve to project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "PS_20174392719_1491204439457_log.csv"

EXPECTED_COLUMNS = [
    "step", "type", "amount", "nameOrig", "oldbalanceOrg", "newbalanceOrig",
    "nameDest", "oldbalanceDest", "newbalanceDest", "isFraud", "isFlaggedFraud",
]


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATA_PATH}. "
            "Download it with: kaggle datasets download -d ealaxi/paysim1 -p data/raw"
        )

    # nrows=100000 keeps this check fast; full-file validation happens in Phase 2 EDA
    df = pd.read_csv(DATA_PATH, nrows=100_000)

    missing = set(EXPECTED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Schema mismatch — missing columns: {missing}")

    print("Environment OK")
    print(f"Sample shape: {df.shape}")
    print(f"Fraud rate in sample: {df['isFraud'].mean():.4%}")
    print(f"Transaction types present: {sorted(df['type'].unique())}")
    print("Phase 1 setup verified.")


if __name__ == "__main__":
    main()