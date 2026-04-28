"""
Builds prompts from dataset examples and parses answers from LLM responses.
"""

import re
from prompts.templates import TEMPLATES, SYSTEM_PROMPT

CHOICE_LABELS = ["A", "B", "C", "D", "E"]


def format_choices(choices: list) -> str:
    """Format a list of answer strings as labelled choices."""
    return "\n".join(f"{CHOICE_LABELS[i]}. {c}" for i, c in enumerate(choices))


def build_prompt(example: dict, condition: str) -> tuple[str, str]:
    """Return (system_prompt, user_prompt) for a given example and condition.

    Args:
        example:   A single ScienceQA dataset row (dict).
        condition: One of "C1", "C2", "C3", "C4".

    Returns:
        A tuple of (system_prompt, user_prompt) strings ready for the API.
    """
    raw_hint = example.get("hint") or ""
    # Only inject a Hint line when the example actually has one.
    # For C1/C3 the {hint_section} placeholder is not in the template, so this is ignored.
    hint_section = f"Hint: {raw_hint}\n\n" if raw_hint else ""

    fields = {
        "lecture":      example.get("lecture") or "No lecture provided.",
        "hint_section": hint_section,
        "grade":        example.get("grade")   or "K-12",
        "question":     example["question"],
        "choices_str":  format_choices(example["choices"]),
    }
    user_prompt = TEMPLATES[condition].format(**fields)
    return SYSTEM_PROMPT, user_prompt


def parse_answer(response_text: str, num_choices: int) -> int | None:
    """Extract the 0-based answer index from the model's response text.

    Tries two strategies:
      1. Look for an explicit "Answer: X" pattern (preferred).
      2. Fall back to a standalone letter at the start of any line.

    Returns None if no valid answer letter is found.
    """
    valid_labels = CHOICE_LABELS[:num_choices]

    # Strategy 1: explicit answer declaration
    pattern = r'\bAnswer\s*:\s*([A-E])\b'
    match = re.search(pattern, response_text, re.IGNORECASE)
    if match:
        letter = match.group(1).upper()
        if letter in valid_labels:
            return ord(letter) - ord('A')

    # Strategy 2: first standalone letter at line start
    for line in response_text.splitlines():
        line = line.strip()
        if line and line[0].upper() in valid_labels:
            if len(line) == 1 or line[1] in (' ', '.', ')', ':'):
                return ord(line[0].upper()) - ord('A')

    return None
