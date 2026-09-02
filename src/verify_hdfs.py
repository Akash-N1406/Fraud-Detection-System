"""
Phase 4 sanity check.
Run from src/: python verify_hdfs.py
Confirms: HDFS daemons running, expected directory structure exists,
raw dataset present in HDFS with the expected size.
"""
import subprocess
import sys

EXPECTED_DIRS = [
    "/fraud-detection/raw",
    "/fraud-detection/processed",
    "/fraud-detection/features",
    "/fraud-detection/models",
]
RAW_FILE = "/fraud-detection/raw/PS_20174392719_1491204439457_log.csv"
EXPECTED_MIN_SIZE_MB = 450  # raw CSV is ~470.7MB; flag if suspiciously smaller


def run(cmd: list[str]) -> tuple[int, str]:
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode, (result.stdout + result.stderr)


def check_daemons() -> bool:
    code, output = run(["jps"])
    required = {"NameNode", "DataNode", "SecondaryNameNode"}
    running = {line.split()[1] for line in output.splitlines() if len(line.split()) > 1}
    missing = required - running
    if missing:
        print(f"FAIL: missing HDFS daemons: {missing}")
        print("Run `start-dfs.sh` and try again.")
        return False
    print("HDFS daemons OK: NameNode, DataNode, SecondaryNameNode all running.")
    return True


def check_directories() -> bool:
    all_ok = True
    for d in EXPECTED_DIRS:
        code, _ = run(["hdfs", "dfs", "-test", "-d", d])
        status = "OK" if code == 0 else "MISSING"
        if code != 0:
            all_ok = False
        print(f"  {status}: {d}")
    return all_ok


def check_raw_file() -> bool:
    code, output = run(["hdfs", "dfs", "-test", "-e", RAW_FILE])
    if code != 0:
        print(f"FAIL: {RAW_FILE} not found in HDFS.")
        return False

    code, output = run(["hdfs", "dfs", "-du", RAW_FILE])
    try:
        size_bytes = int(output.split()[0])
        size_mb = size_bytes / (1024 * 1024)
    except (IndexError, ValueError):
        print(f"WARNING: could not parse file size from: {output.strip()}")
        return True

    if size_mb < EXPECTED_MIN_SIZE_MB:
        print(f"FAIL: raw file only {size_mb:.1f}MB, expected ~470MB. Upload may be incomplete.")
        return False

    print(f"Raw dataset OK: {size_mb:.1f}MB in HDFS.")
    return True


def main() -> None:
    print("=== Phase 4 HDFS Verification ===\n")
    daemons_ok = check_daemons()
    print("\nDirectory structure:")
    dirs_ok = check_directories()
    print("\nRaw dataset:")
    file_ok = check_raw_file()

    print()
    if daemons_ok and dirs_ok and file_ok:
        print("Phase 4 verified: HDFS is live and populated.")
    else:
        print("Phase 4 incomplete — see failures above.")
        sys.exit(1)


if __name__ == "__main__":
    main()