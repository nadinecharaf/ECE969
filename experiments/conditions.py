"""
Defines the five experimental input configurations (C0–C4).
"""

CONDITIONS: dict[str, dict] = {
    "C0": {
        "label":        "Question Only (No Context)",
        "uses_lecture": False,
        "uses_hint":    False,
        "uses_grade":   False,
    },
    "C1": {
        "label":        "Baseline (Lecture Only)",
        "uses_lecture": True,
        "uses_hint":    False,
        "uses_grade":   False,
    },
    "C2": {
        "label":        "+ Pedagogical Hint",
        "uses_lecture": True,
        "uses_hint":    True,
        "uses_grade":   False,
    },
    "C3": {
        "label":        "+ Grade Context",
        "uses_lecture": True,
        "uses_hint":    False,
        "uses_grade":   True,
    },
    "C4": {
        "label":        "Full Context (All Info)",
        "uses_lecture": True,
        "uses_hint":    True,
        "uses_grade":   True,
    },
}
