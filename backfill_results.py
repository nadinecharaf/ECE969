"""
Back-fill lecture, hint, and solution into result files that were saved
before runner.py was updated to include those fields.

Safe to run multiple times — skips files that already have the fields.

Usage:
    python backfill_results.py
"""

import json
from pathlib import Path
from data.loader import load_split
import config


def build_lookup(dataset) -> dict[str, dict]:
    """Build a question-text → example dict for fast lookup."""
    return {ex["question"]: ex for ex in dataset}


def backfill(results_dir: Path = config.RESULTS_DIR):
    json_files = sorted(results_dir.glob("*.json"))
    if not json_files:
        print("No result files found.")
        return

    # Load once — all splits share the same questions in the test set
    print("Loading test split for lookup...")
    lookup = build_lookup(load_split("test"))
    print(f"  {len(lookup)} questions indexed.")

    for path in json_files:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        results = data.get("results", [])
        if not results:
            continue

        # Check if already back-filled
        if "lecture" in results[0]:
            print(f"  SKIP (already has lecture): {path.name}")
            continue

        n_filled = 0
        for row in results:
            ex = lookup.get(row["question"])
            if ex:
                row["lecture"]  = ex.get("lecture")  or ""
                row["hint"]     = ex.get("hint")      or ""
                row["solution"] = ex.get("solution")  or ""
                n_filled += 1

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        print(f"  UPDATED ({n_filled}/{len(results)} rows): {path.name}")

    print("Done.")


if __name__ == "__main__":
    backfill()
