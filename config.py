import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Dataset ───────────────────────────────────────────────────────────────────
DATASET_PATH = Path(
    "C:/Users/alisa/Datasets/ScienceQA/derek-thomas___science_qa"
    "/default/0.0.0/f18b0a70359ebfb41f658fd564208d0355b013f4"
)

# ── Sampling ──────────────────────────────────────────────────────────────────
SAMPLE_SIZE        = 375          # 75 per grade × 5 grades (grade3–grade7)
SAMPLES_PER_GRADE  = 75
TARGET_GRADES      = ["grade3", "grade4", "grade5", "grade6", "grade7"]
RANDOM_SEED        = 42

# ── Generation settings (shared across all models) ───────────────────────────
MAX_TOKENS  = 300   # answer line (~5 tokens) + brief explanation (~2 sentences)
TEMPERATURE = 0.0       # deterministic — set to 0.1 for Ollama if output is too repetitive

# ── Local models (Ollama) ─────────────────────────────────────────────────────
OLLAMA_MODEL = "llama3.2:3b"            # default local model
OLLAMA_HOST  = "http://localhost:11434"  # Ollama server address (default)

# Available local models (add more after ollama pull <name>)
OLLAMA_MODELS = [
    "llama3.2:3b",    # 2.0 GB  — small Llama, fast baseline
    "phi4-mini",      # 2.5 GB  — Microsoft small reasoning model
    "llama3.1:8b",    # 4.9 GB  — medium Llama
    "mixtral:8x7b",   # 26 GB   — Mixtral MoE, strongest local
]

# ── Cloud models (API keys required) ─────────────────────────────────────────
CLAUDE_MODEL  = "claude-haiku-4-5-20251001"
OPENAI_MODEL  = "gpt-4o-mini"

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
OPENAI_API_KEY    = os.getenv("OPENAI_API_KEY")

# ── Groq (fast cloud inference, OpenAI-compatible) ────────────────────────────
GROQ_API_KEY          = os.getenv("GROQ_API_KEY")
GROQ_INTER_CALL_DELAY = 2.5   # seconds between requests — avoids token/min rate limit

# ── Google Gemini (free via AI Studio — 1,500 req/day per account) ───────────
# Add up to 2 keys (2 Google accounts) for 3,000 req/day total
GEMINI_API_KEYS = [k for k in [
    os.getenv("GEMINI_API_KEY_1"),
    os.getenv("GEMINI_API_KEY_2"),
] if k]

GEMINI_MODELS = [
    "gemini-2.0-flash-lite",   # free tier — lite version
    "gemini-1.5-flash",        # legacy name — not available in v1beta
    "gemini-2.0-flash",        # NOT free tier (limit=0) — requires billing
]
GEMINI_INTER_CALL_DELAY = 5.0   # seconds between requests — free tier: 15 RPM max

# ── GitHub Models (OpenAI-compatible, free with GitHub Student Pack) ──────────
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
GITHUB_MODELS_BASE_URL = "https://models.inference.ai.azure.com"

GITHUB_MODELS = [
    "gpt-4o",                    # OpenAI GPT-4o via GitHub
    "gpt-4o-mini",               # OpenAI GPT-4o-mini via GitHub
    "Llama-3.3-70B-Instruct",    # Meta Llama 3.3 70B via GitHub
    "Phi-4",                     # Microsoft Phi-4 via GitHub
]

GROQ_MODELS = [
    "llama-3.3-70b-versatile",   # strongest Groq model — Llama 3.3 70B
    "llama-3.1-8b-instant",      # fast small model — matches local llama3.1:8b
    "mixtral-8x7b-32768",        # Mixtral MoE — matches local mixtral:8x7b
]

# ── All models to run in a full study ─────────────────────────────────────────
# Comment/uncomment as you add API keys
STUDY_MODELS = [
    "llama3.2:3b",                # local — no key needed
    "phi4-mini",                  # local — no key needed
    "llama3.1:8b",                # local — no key needed
    "mixtral:8x7b",               # local — no key needed
    "claude-haiku-4-5-20251001",  # Anthropic — needs ANTHROPIC_API_KEY
    "gemini-2.0-flash-lite",      # Google — needs GEMINI_API_KEY_1 (+ optional KEY_2)
]

# ── Output ────────────────────────────────────────────────────────────────────
RESULTS_DIR = Path("results")
