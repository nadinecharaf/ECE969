"""
Main experiment runner.

Usage examples:
    # Run all conditions with the default local model
    python -m experiments.runner

    # Run all conditions with a specific model
    python -m experiments.runner --model phi4-mini

    # Run only C1 and C3 with llama
    python -m experiments.runner --model llama3.2:3b --cond C1 C3

    # Run all models defined in config.STUDY_MODELS (full study)
    python -m experiments.runner --all-models
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

from tqdm import tqdm

import config
from data.loader import load_split
from data.sampler import stratified_sample
from experiments.conditions import CONDITIONS
from models.factory import get_client
from models.base import LLMClient
from prompts.builder import build_prompt, parse_answer


def run_condition(
    condition: str,
    dataset,
    client: LLMClient,
    results_dir: Path = config.RESULTS_DIR,
    inter_call_delay: float = 0.1,
) -> list[dict]:
    """Run one condition with one model. Saves results and returns rows."""

    results_dir.mkdir(parents=True, exist_ok=True)
    timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_slug = client.model_id.replace("/", "-")
    out_file   = results_dir / f"{condition}_{model_slug}_{timestamp}.json"

    results: list[dict] = []
    n_correct = 0

    label = CONDITIONS[condition]["label"]
    print(f"\n{'-' * 65}")
    print(f"  Condition : {condition} - {label}")
    print(f"  Model     : {client.model_id}")
    print(f"  Examples  : {len(dataset)}")
    print(f"{'-' * 65}")

    for example in tqdm(dataset, desc=f"{condition}/{client.model_id}"):
        system_prompt, user_prompt = build_prompt(example, condition)
        response  = client.generate(system_prompt, user_prompt)
        predicted = parse_answer(response, len(example["choices"]))
        gold      = int(example["answer"])
        correct   = predicted == gold

        if correct:
            n_correct += 1

        results.append({
            "model":            client.model_id,
            "condition":        condition,
            "question":         example["question"],
            "choices":          example["choices"],
            "gold_answer":      gold,
            "predicted_answer": predicted,
            "is_correct":       correct,
            "grade":            example.get("grade"),
            "subject":          example.get("subject"),
            "topic":            example.get("topic"),
            "skill":            example.get("skill"),
            "lecture":          example.get("lecture")  or "",
            "hint":             example.get("hint")     or "",
            "solution":         example.get("solution") or "",
            "has_lecture":      bool(example.get("lecture")),
            "has_hint":         bool(example.get("hint")),
            "has_solution":     bool(example.get("solution")),
            "response":         response,
        })

        time.sleep(inter_call_delay)

    accuracy = n_correct / len(dataset) if dataset else 0.0
    print(f"  Accuracy  : {n_correct}/{len(dataset)} = {accuracy:.3f}")

    payload = {
        "model":     client.model_id,
        "condition": condition,
        "label":     label,
        "n":         len(dataset),
        "accuracy":  accuracy,
        "results":   results,
    }
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"  Saved     : {out_file}")

    return results


def run_model(
    model_name: str,
    sample,
    conditions: list[str],
) -> dict[str, list[dict]]:
    """Run all requested conditions for a single model."""
    client = get_client(model_name)
    return {cond: run_condition(cond, sample, client) for cond in conditions}


def run_study(
    models: list[str],
    conditions: list[str],
) -> dict[str, dict[str, list[dict]]]:
    """Run the full study: every model × every condition on the same sample."""

    print("Loading test split...")
    test_ds = load_split("test")
    print(f"Sampling {config.SAMPLES_PER_GRADE} examples x "
          f"{len(config.TARGET_GRADES)} grades "
          f"({', '.join(config.TARGET_GRADES)})...")
    sample = stratified_sample(test_ds)
    print(f"Final sample size : {len(sample)}")
    print(f"Models            : {models}")
    print(f"Conditions        : {conditions}")

    all_results: dict[str, dict[str, list[dict]]] = {}
    for model_name in models:
        all_results[model_name] = run_model(model_name, sample, conditions)

    _print_summary(all_results, conditions)
    return all_results


def _print_summary(
    all_results: dict[str, dict[str, list[dict]]],
    conditions: list[str],
) -> None:
    from evaluation.metrics import answer_accuracy

    print(f"\n{'=' * 65}")
    print(f"  {'Model':<22} {'Cond':<6} {'N':>5} {'Accuracy':>10}")
    print(f"{'-' * 65}")
    for model_name, cond_results in sorted(all_results.items()):
        for cond in conditions:
            res = cond_results.get(cond, [])
            print(f"  {model_name:<22} {cond:<6} {len(res):>5} {answer_accuracy(res):>10.3f}")
    print(f"{'=' * 65}\n")


# ── CLI entry point ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ScienceQA ablation study.")
    parser.add_argument(
        "--model", nargs="+", default=None,
        help="Model name(s) to run (e.g. llama3.2:3b phi4-mini). "
             "Defaults to config.STUDY_MODELS.",
    )
    parser.add_argument(
        "--all-models", action="store_true",
        help="Run all models listed in config.STUDY_MODELS.",
    )
    parser.add_argument(
        "--cond", nargs="+", choices=list(CONDITIONS.keys()), default=None,
        help="Conditions to run (default: all four).",
    )
    args = parser.parse_args()

    models     = config.STUDY_MODELS if args.all_models else (args.model or config.STUDY_MODELS)
    conditions = args.cond or list(CONDITIONS.keys())

    run_study(models=models, conditions=conditions)
