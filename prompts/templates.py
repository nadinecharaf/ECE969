"""
Prompt templates for the five experimental conditions (C0–C4).

Each template is a plain Python format-string. The variables available are:
  {lecture}       – background lecture passage   (C1–C4 only)
  {hint_section}  – "Hint: <text>\n\n" or ""     (C2, C4 only)
  {grade}         – grade level string            (C3, C4 only)
  {question}      – the question text
  {choices_str}   – pre-formatted "A. ...\nB. ..." string
"""

SYSTEM_PROMPT = (
    "You are a helpful science tutor for K-12 students. "
    "Answer the multiple-choice question carefully and explain your reasoning clearly."
)

# ── C0: Question Only – no context ────────────────────────────────────────────
C0_TEMPLATE = """\
Question: {question}

Choices:
{choices_str}

Instructions:
- Write your answer on the first line as exactly: Answer: <letter>  (e.g. Answer: A)
- Then write a brief explanation.

Your response:"""

# ── C1: Baseline – Lecture only ───────────────────────────────────────────────
C1_TEMPLATE = """\
Background Lecture:
{lecture}

Question: {question}

Choices:
{choices_str}

Instructions:
- Write your answer on the first line as exactly: Answer: <letter>  (e.g. Answer: A)
- Then write a brief explanation grounded in the lecture above.

Your response:"""

# ── C2: + Pedagogical Hint ────────────────────────────────────────────────────
C2_TEMPLATE = """\
Background Lecture:
{lecture}

{hint_section}Question: {question}

Choices:
{choices_str}

Instructions:
- Write your answer on the first line as exactly: Answer: <letter>  (e.g. Answer: A)
- Then write a brief explanation grounded in the lecture above.

Your response:"""

# ── C3: + Grade Context ───────────────────────────────────────────────────────
C3_TEMPLATE = """\
Background Lecture:
{lecture}

Grade Level: {grade}

Question: {question}

Choices:
{choices_str}

Instructions:
- Write your answer on the first line as exactly: Answer: <letter>  (e.g. Answer: A)
- Then write a brief explanation grounded in the lecture above, using language
  appropriate for a {grade} student.

Your response:"""

# ── C4: Full Context – Lecture + Hint + Grade ─────────────────────────────────
C4_TEMPLATE = """\
Background Lecture:
{lecture}

{hint_section}Grade Level: {grade}

Question: {question}

Choices:
{choices_str}

Instructions:
- Write your answer on the first line as exactly: Answer: <letter>  (e.g. Answer: A)
- Then write a brief explanation grounded in the lecture above, using language
  appropriate for a {grade} student.

Your response:"""

TEMPLATES: dict[str, str] = {
    "C0": C0_TEMPLATE,
    "C1": C1_TEMPLATE,
    "C2": C2_TEMPLATE,
    "C3": C3_TEMPLATE,
    "C4": C4_TEMPLATE,
}
