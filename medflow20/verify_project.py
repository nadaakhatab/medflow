"""Local verification entry point for the MedFlow project.

Run this from the project root after activating the project environment:
    python verify_project.py

It runs all Day 1-4 unit/compliance suites and audits the separate frozen Day 2 index used by Day 4.
It does not rebuild or delete any vector database.
"""
from __future__ import annotations

import subprocess
import sys


def run(cmd):
    print("\n$", " ".join(cmd), flush=True)
    return subprocess.call(cmd)


def main() -> int:
    commands = [
        [sys.executable, "-m", "unittest", "discover", "-s", "day1", "-p", "test_*.py", "-v"],
        [sys.executable, "-m", "unittest", "discover", "-s", "day2", "-p", "test_*.py", "-v"],
        [sys.executable, "-m", "unittest", "discover", "-s", "day3", "-p", "test_*.py", "-v"],
        [sys.executable, "-m", "unittest", "discover", "-s", "day4", "-p", "test_*.py", "-v"],
        [sys.executable, "day4/evaluate_day4.py", "--audit-index", "--persist-dir", "chroma_db_day2_frozen", "--collection", "thyroid_day2_frozen"],
    ]
    failed = 0
    for cmd in commands:
        failed += 1 if run(cmd) != 0 else 0
    if failed:
        print(f"\nVerification finished with {failed} failing command(s).")
        return 1
    print("\nAll source test suites and the audited frozen Day 2 index check completed successfully.")
    print("For full Day 4 metrics, follow day4/README.md and evaluate an audited frozen index.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
