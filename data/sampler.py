"""
Sampling strategy for the ablation study.

Eligibility rules (all must hold):
  - No image          — text-only models; images are never passed to the LLM
  - Has lecture       — required by C1–C4 (non-empty)
  - Has hint          — required by C2/C4; guarantees hint_section is never empty
  - Has grade         — required by C3/C4 and grade-calibration metric
  - Has solution      — used in post-hoc analysis / metrics
  - Grade in TARGET   — only grades 3–7 (grade3, grade4, grade5, grade6, grade7)

Sampling:
  Exactly SAMPLES_PER_GRADE (75) examples drawn uniformly at random from each
  of the five target grades → 375 examples total per experiment run.

Grade coverage rationale:
  Grades 3–7 all have ≥ 75 qualifying examples in the test split.
  The selection enables close comparisons (3 vs 4, 4 vs 5, 5 vs 6, 6 vs 7)
  and far comparisons (3 vs 7, 3 vs 6, etc.).
  Grades 1–2 and 8–12 are excluded due to insufficient qualifying examples.
"""

import random
from datasets import Dataset
import config


def is_eligible(example: dict) -> bool:
    """Return True if the example passes all inclusion criteria."""
    if example["image"] is not None:
        return False                         # text-only models only
    if not (example.get("lecture")  or "").strip():
        return False                         # lecture required for C1-C4
    if not (example.get("hint")     or "").strip():
        return False                         # hint required for C2/C4
    if not (example.get("grade")    or "").strip():
        return False                         # grade required for C3/C4 and metrics
    if not (example.get("solution") or "").strip():
        return False                         # solution required for analysis
    if example.get("grade") not in config.TARGET_GRADES:
        return False                         # only grades 3-7
    return True


def stratified_sample(
    dataset: Dataset,
    samples_per_grade: int = config.SAMPLES_PER_GRADE,
    seed: int = config.RANDOM_SEED,
) -> Dataset:
    """Return exactly samples_per_grade examples from each TARGET_GRADE.

    Args:
        dataset:          Source HuggingFace Dataset (usually the test split).
        samples_per_grade: Number of examples to draw per grade (default: 75).
        seed:             Random seed for reproducibility.

    Returns:
        A Dataset of size len(TARGET_GRADES) * samples_per_grade (= 375).

    Raises:
        ValueError: if any target grade has fewer than samples_per_grade examples.
    """
    rng = random.Random(seed)

    # ── 1. Apply eligibility filter ───────────────────────────────────────────
    eligible_idx = [i for i, ex in enumerate(dataset) if is_eligible(ex)]
    dataset = dataset.select(eligible_idx)

    # ── 2. Group indices by grade ─────────────────────────────────────────────
    grade_idx: dict[str, list[int]] = {g: [] for g in config.TARGET_GRADES}
    for i, ex in enumerate(dataset):
        g = ex.get("grade")
        if g in grade_idx:
            grade_idx[g].append(i)

    # ── 3. Validate availability ──────────────────────────────────────────────
    for g, idxs in grade_idx.items():
        if len(idxs) < samples_per_grade:
            raise ValueError(
                f"Grade {g} has only {len(idxs)} eligible examples, "
                f"need {samples_per_grade}."
            )

    # ── 4. Draw exactly samples_per_grade from each grade ────────────────────
    selected: list[int] = []
    for g in config.TARGET_GRADES:
        selected.extend(rng.sample(grade_idx[g], samples_per_grade))

    return dataset.select(sorted(selected))
