"""
Quantitative evaluation metrics (proposal §Evaluation Metrics).

Metrics
-------
1.  answer_accuracy          – fraction of correct predictions
2.  choice_consistency       – fraction of parseable answers
3.  accuracy_by_grade        – per-grade accuracy dict
4.  lecture_grounding        – how much the response draws on the lecture
                               (semantic cosine + ROUGE-1 keyword overlap)
5.  hint_alignment           – how much the response draws on the hint
                               (semantic cosine + ROUGE-1 keyword overlap)
6.  grade_calibration        – Flesch-Kincaid grade of the response
                               vs actual numeric grade (calibration error)
7.  question_difficulty      – per-grade readability stats on the question text
"""

import re
from collections import defaultdict


# ── Basic accuracy metrics ─────────────────────────────────────────────────────

def answer_accuracy(results: list[dict]) -> float:
    """Fraction of examples where predicted == gold answer."""
    if not results:
        return 0.0
    return sum(1 for r in results if r.get("is_correct")) / len(results)


def choice_consistency(results: list[dict]) -> float:
    """Fraction of examples where the model produced a parseable answer letter."""
    if not results:
        return 0.0
    parseable = sum(1 for r in results if r.get("predicted_answer") is not None)
    return parseable / len(results)


def accuracy_by_grade(results: list[dict]) -> dict[str, float]:
    """Per-grade accuracy (supports RQ3: grade-level conditioning effect)."""
    correct: dict[str, int] = defaultdict(int)
    total:   dict[str, int] = defaultdict(int)

    for r in results:
        grade = r.get("grade") or "unknown"
        total[grade] += 1
        if r.get("is_correct"):
            correct[grade] += 1

    return {g: correct[g] / total[g] for g in sorted(total)}


def condition_summary(all_results: dict[str, list[dict]]) -> list[dict]:
    """Build a summary row per condition — useful for tables / DataFrames."""
    rows = []
    for cond, results in sorted(all_results.items()):
        rows.append({
            "condition":   cond,
            "n":           len(results),
            "accuracy":    answer_accuracy(results),
            "consistency": choice_consistency(results),
        })
    return rows


# ── Text preprocessing helper ──────────────────────────────────────────────────

_STOPWORDS = frozenset({
    "a", "an", "the", "is", "it", "in", "of", "to", "and", "or", "for",
    "on", "at", "by", "with", "as", "be", "was", "are", "were", "this",
    "that", "from", "not", "but", "so", "if", "do", "does", "did", "has",
    "have", "had", "can", "will", "would", "could", "should", "may", "might",
    "its", "their", "they", "we", "i", "you", "he", "she", "what", "which",
    "when", "where", "how", "all", "into", "than", "more", "also", "been",
})


def _tokens(text: str) -> list[str]:
    """Lowercase alpha tokens, stopwords removed."""
    return [
        w for w in re.findall(r"[a-z]+", text.lower())
        if w not in _STOPWORDS and len(w) > 2
    ]


def _rouge1_recall(reference: str, hypothesis: str) -> float:
    """ROUGE-1 recall: fraction of reference tokens found in hypothesis."""
    ref_tokens = set(_tokens(reference))
    hyp_tokens = set(_tokens(hypothesis))
    if not ref_tokens:
        return 0.0
    return len(ref_tokens & hyp_tokens) / len(ref_tokens)


# ── Sentence-transformers singleton ───────────────────────────────────────────

_ST_MODEL = None


def _get_st_model():
    global _ST_MODEL
    if _ST_MODEL is None:
        from sentence_transformers import SentenceTransformer
        _ST_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    return _ST_MODEL


def _cosine(a, b) -> float:
    """Cosine similarity between two 1-D numpy arrays."""
    import numpy as np
    a, b = np.array(a, dtype=float), np.array(b, dtype=float)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom > 0 else 0.0


def _batch_cosine_sim(texts_a: list[str], texts_b: list[str]) -> list[float]:
    """Batch cosine similarity using sentence-transformers embeddings."""
    model = _get_st_model()
    embs_a = model.encode(texts_a, convert_to_numpy=True, show_progress_bar=False)
    embs_b = model.encode(texts_b, convert_to_numpy=True, show_progress_bar=False)
    return [_cosine(a, b) for a, b in zip(embs_a, embs_b)]


# ── Lecture grounding ─────────────────────────────────────────────────────────

def lecture_grounding(results: list[dict]) -> dict:
    """How much does the model response draw on the provided lecture?

    Returns:
        {
          "semantic": float,   # mean cosine similarity (sentence-transformers)
          "overlap":  float,   # mean ROUGE-1 recall (keyword overlap)
          "n":        int,     # number of rows with non-empty lecture
        }
    """
    rows = [r for r in results if r.get("lecture") and r.get("response")]
    if not rows:
        return {"semantic": 0.0, "overlap": 0.0, "n": 0}

    lectures  = [r["lecture"]  for r in rows]
    responses = [r["response"] for r in rows]

    sims     = _batch_cosine_sim(responses, lectures)
    overlaps = [_rouge1_recall(lec, resp) for lec, resp in zip(lectures, responses)]

    return {
        "semantic": sum(sims) / len(sims),
        "overlap":  sum(overlaps) / len(overlaps),
        "n":        len(rows),
    }


# ── Hint alignment ────────────────────────────────────────────────────────────

def hint_alignment(results: list[dict]) -> dict:
    """How much does the model response draw on the provided hint?

    Returns:
        {
          "semantic": float,
          "overlap":  float,
          "n":        int,     # number of rows with non-empty hint
        }
    """
    rows = [r for r in results if r.get("hint") and r.get("response")]
    if not rows:
        return {"semantic": 0.0, "overlap": 0.0, "n": 0}

    hints     = [r["hint"]     for r in rows]
    responses = [r["response"] for r in rows]

    sims     = _batch_cosine_sim(responses, hints)
    overlaps = [_rouge1_recall(hint, resp) for hint, resp in zip(hints, responses)]

    return {
        "semantic": sum(sims) / len(sims),
        "overlap":  sum(overlaps) / len(overlaps),
        "n":        len(rows),
    }


# ── Grade calibration ─────────────────────────────────────────────────────────

def _grade_to_num(grade_str: str) -> int | None:
    """Convert 'grade3' or 'grade 3' to integer 3. Returns None if unparseable."""
    m = re.search(r"(\d+)", str(grade_str))
    return int(m.group(1)) if m else None


def _word_count(text: str) -> int:
    """Number of whitespace-separated tokens in text."""
    return len(text.split())


def grade_calibration(results: list[dict]) -> dict:
    """Compare Flesch-Kincaid reading level and response length to actual grade.

    Higher grades should produce longer, more complex responses when grade
    metadata is provided (C3/C4). Response length (word count) is included
    alongside FKGL to capture the detail dimension separately from complexity.

    Returns:
        {
          "mean_fkgl":          float,  # mean FK grade of all responses
          "mean_grade":         float,  # mean numeric grade in sample
          "mean_response_len":  float,  # mean response word count
          "calibration_error":  float,  # mean |FKGL(response) - grade_num|
          "by_grade": {
              "grade3": {
                  "fkgl":         float,
                  "expected":     int,
                  "error":        float,
                  "response_len": float,  # mean word count for this grade
              }, ...
          }
        }
    """
    import textstat

    fkgl_vals, grade_nums, errors, lengths = [], [], [], []
    by_grade: dict[str, list] = defaultdict(list)

    for r in results:
        response  = (r.get("response") or "").strip()
        grade_str = r.get("grade") or ""
        grade_num = _grade_to_num(grade_str)
        if not response or grade_num is None:
            continue

        fk  = textstat.flesch_kincaid_grade(response)
        wc  = _word_count(response)
        fkgl_vals.append(fk)
        grade_nums.append(grade_num)
        errors.append(abs(fk - grade_num))
        lengths.append(wc)
        by_grade[grade_str].append((fk, grade_num, wc))

    if not fkgl_vals:
        return {
            "mean_fkgl": 0.0, "mean_grade": 0.0,
            "mean_response_len": 0.0, "calibration_error": 0.0, "by_grade": {},
        }

    per_grade_summary = {}
    for g, triples in sorted(by_grade.items()):
        fks  = [t[0] for t in triples]
        exp  = triples[0][1]
        wcs  = [t[2] for t in triples]
        per_grade_summary[g] = {
            "fkgl":         sum(fks) / len(fks),
            "expected":     exp,
            "error":        sum(abs(f - exp) for f in fks) / len(fks),
            "response_len": sum(wcs) / len(wcs),
        }

    return {
        "mean_fkgl":         sum(fkgl_vals) / len(fkgl_vals),
        "mean_grade":        sum(grade_nums) / len(grade_nums),
        "mean_response_len": sum(lengths) / len(lengths),
        "calibration_error": sum(errors) / len(errors),
        "by_grade":          per_grade_summary,
    }


# ── Question difficulty ───────────────────────────────────────────────────────

def _type_token_ratio(text: str) -> float:
    """Lexical diversity: unique tokens / total tokens."""
    tokens = _tokens(text)
    if not tokens:
        return 0.0
    return len(set(tokens)) / len(tokens)


def _avg_word_len(text: str) -> float:
    words = re.findall(r"[a-zA-Z]+", text)
    if not words:
        return 0.0
    return sum(len(w) for w in words) / len(words)


def question_difficulty(results: list[dict]) -> dict:
    """Per-grade readability of the question text.

    Metrics per grade:
      - fkgl:      Flesch-Kincaid Grade Level of the question
      - avg_wlen:  average word length (proxy for vocabulary complexity)
      - ttr:       type-token ratio (lexical diversity)
      - n:         number of questions

    Reveals whether harder grades genuinely have more complex questions.
    """
    import textstat

    by_grade: dict[str, list[dict]] = defaultdict(list)

    for r in results:
        question  = (r.get("question") or "").strip()
        grade_str = r.get("grade") or "unknown"
        if not question:
            continue

        by_grade[grade_str].append({
            "fkgl":     textstat.flesch_kincaid_grade(question),
            "avg_wlen": _avg_word_len(question),
            "ttr":      _type_token_ratio(question),
        })

    summary = {}
    for g, items in sorted(by_grade.items()):
        n = len(items)
        summary[g] = {
            "fkgl":     sum(x["fkgl"]     for x in items) / n,
            "avg_wlen": sum(x["avg_wlen"] for x in items) / n,
            "ttr":      sum(x["ttr"]      for x in items) / n,
            "n":        n,
        }
    return summary
