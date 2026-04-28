"""
Load saved result files and compute / display aggregate statistics.

Usage (from project root):
    python -m evaluation.analyzer
    python -m evaluation.analyzer --rich      # include semantic + overlap metrics
"""

import json
from pathlib import Path

import pandas as pd

import config
from evaluation.metrics import (
    answer_accuracy,
    choice_consistency,
    accuracy_by_grade,
    condition_summary,
    lecture_grounding,
    hint_alignment,
    grade_calibration,
    question_difficulty,
)
from experiments.conditions import CONDITIONS


def load_results(results_dir: Path = config.RESULTS_DIR) -> dict[str, list[dict]]:
    """Load the latest result file for each (condition, model) pair found in results_dir."""
    all_results: dict[str, list[dict]] = {}

    # Files are named {COND}_{model}_{timestamp}.json — sort so last = most recent
    for json_file in sorted(results_dir.glob("*.json")):
        with open(json_file, encoding="utf-8") as f:
            data = json.load(f)
        cond  = data.get("condition")
        model = data.get("model", "unknown")
        if cond:
            key = f"{cond}|{model}"
            all_results[key] = data["results"]   # overwrite -> keeps latest run

    return all_results


def to_dataframe(results: list[dict]) -> pd.DataFrame:
    """Convert a results list to a DataFrame (drops raw 'response' column)."""
    rows = [{k: v for k, v in r.items() if k != "response"} for r in results]
    return pd.DataFrame(rows)


def print_summary(all_results: dict[str, list[dict]]) -> None:
    """Print a formatted accuracy / consistency table to stdout."""
    width = 72
    print(f"\n{'=' * width}")
    print(f"  {'Key':<28} {'N':>5} {'Accuracy':>10} {'Consistency':>12}")
    print(f"{'-' * width}")
    for key in sorted(all_results):
        results = all_results[key]
        cond, model = key.split("|", 1)
        label = CONDITIONS.get(cond, {}).get("label", "")
        print(
            f"  {key:<28} {len(results):>5} {answer_accuracy(results):>10.3f}"
            f" {choice_consistency(results):>12.3f}  {label}"
        )
    print(f"{'=' * width}\n")


def print_grade_breakdown(all_results: dict[str, list[dict]]) -> None:
    """Print per-grade accuracy for each condition+model key."""
    for key, results in sorted(all_results.items()):
        cond, model = key.split("|", 1)
        print(f"\n  {key} -- {CONDITIONS.get(cond, {}).get('label', '')}")
        for grade, acc in accuracy_by_grade(results).items():
            bar = "#" * int(acc * 20)
            print(f"    {grade:<12} {acc:.3f}  {bar}")


def print_rich_metrics(all_results: dict[str, list[dict]]) -> None:
    """Print lecture grounding and hint alignment for each condition+model."""
    width = 72
    print(f"\n{'=' * width}")
    print("  Rich Metrics (require sentence-transformers + textstat)")
    print(f"{'-' * width}")
    print(f"  {'Key':<28} {'LG-sem':>7} {'LG-ovlp':>8} {'HA-sem':>7} {'HA-ovlp':>8}")
    print(f"{'-' * width}")

    for key in sorted(all_results):
        results = all_results[key]
        lg = lecture_grounding(results)
        ha = hint_alignment(results)
        print(
            f"  {key:<28} {lg['semantic']:>7.3f} {lg['overlap']:>8.3f}"
            f" {ha['semantic']:>7.3f} {ha['overlap']:>8.3f}"
        )
    print(f"{'=' * width}")
    print("  LG = lecture grounding, HA = hint alignment")
    print("  sem = cosine similarity, ovlp = ROUGE-1 recall\n")


def print_grade_calibration(all_results: dict[str, list[dict]]) -> None:
    """Print grade calibration table (response FK grade vs actual grade)."""
    width = 72
    print(f"\n{'=' * width}")
    print("  Grade Calibration (Flesch-Kincaid Grade Level of responses)")
    print(f"{'-' * width}")

    for key in sorted(all_results):
        results = all_results[key]
        gc = grade_calibration(results)
        if not gc["by_grade"]:
            continue
        print(f"\n  {key}  (mean FKGL={gc['mean_fkgl']:.1f}, "
              f"mean grade={gc['mean_grade']:.1f}, "
              f"calib err={gc['calibration_error']:.2f}, "
              f"mean len={gc['mean_response_len']:.0f} words)")
        print(f"    {'Grade':<10} {'Expected':>9} {'FKGL':>7} {'Error':>7} {'Len(words)':>11}")
        for g, vals in gc["by_grade"].items():
            print(f"    {g:<10} {vals['expected']:>9} {vals['fkgl']:>7.1f} "
                  f"{vals['error']:>7.2f} {vals['response_len']:>11.0f}")

    print(f"\n{'=' * width}\n")


def print_question_difficulty(all_results: dict[str, list[dict]]) -> None:
    """Print per-grade question difficulty stats (uses first condition's data)."""
    if not all_results:
        return
    first_key = sorted(all_results)[0]
    results = all_results[first_key]
    qd = question_difficulty(results)

    width = 72
    print(f"\n{'=' * width}")
    print("  Question Difficulty by Grade (question text readability)")
    print(f"{'-' * width}")
    print(f"  {'Grade':<10} {'FKGL':>7} {'AvgWLen':>8} {'TTR':>7} {'N':>5}")
    print(f"{'-' * width}")
    for g, vals in qd.items():
        print(f"  {g:<10} {vals['fkgl']:>7.1f} {vals['avg_wlen']:>8.2f}"
              f" {vals['ttr']:>7.3f} {vals['n']:>5}")
    print(f"{'=' * width}\n")


# ── CLI entry point ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Analyze ScienceQA experiment results.")
    parser.add_argument("--rich", action="store_true",
                        help="Compute and print rich metrics (lecture grounding, "
                             "hint alignment, grade calibration, question difficulty). "
                             "Requires sentence-transformers and textstat.")
    args = parser.parse_args()

    results = load_results()
    if not results:
        print("No result files found in", config.RESULTS_DIR)
    else:
        print_summary(results)
        print_grade_breakdown(results)
        if args.rich:
            print_rich_metrics(results)
            print_grade_calibration(results)
            print_question_difficulty(results)
