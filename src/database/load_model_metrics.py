"""
Phase 9 — loads Phase 6's model comparison results into PostgreSQL.
Run once from src/: python database/load_model_metrics.py

Populates model_metrics from reports/phase6_full_comparison.json so
Phase 11's dashboard can show the model comparison table from real
database rows rather than a static file.
"""
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from database.db import get_connection, insert_model_metrics

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
METRICS_PATH = PROJECT_ROOT / "reports" / "phase6_full_comparison.json"


def main() -> None:
    if not METRICS_PATH.exists():
        raise FileNotFoundError(
            f"{METRICS_PATH} not found. Run Phase 6's "
            "ml/training/train_full_comparison.py first."
        )

    results = json.loads(METRICS_PATH.read_text())

    with get_connection() as conn:
        for result in results:
            insert_model_metrics(conn, result)
            print(f"Loaded: {result['model']}")

    print(f"\n{len(results)} model metrics rows inserted.")


if __name__ == "__main__":
    main()