"""
Model factory — resolves a model name string to the right LLMClient.

Usage:
    from models.factory import get_client
    client = get_client("llama3.2:3b")
    response = client.generate(system, user)

Supported model name patterns:
    "llama3.2:3b"                → OllamaClient       (local)
    "phi4-mini"                  → OllamaClient       (local)
    "llama-3.3-70b-versatile"    → GroqClient         (checked against config.GROQ_MODELS)
    "mixtral-8x7b-32768"         → GroqClient         (checked against config.GROQ_MODELS)
    "gpt-4o-mini"                → GitHubModelsClient (checked against config.GITHUB_MODELS)
    "Llama-3.3-70B-Instruct"     → GitHubModelsClient (checked against config.GITHUB_MODELS)
    "claude-haiku-4-5-..."       → ClaudeClient
    "gpt-4o-mini" (OpenAI)       → OpenAIClient       (only if not in GITHUB_MODELS)

Note: GITHUB_MODELS and GROQ_MODELS are checked BEFORE Ollama/OpenAI because
      their model names overlap with other provider prefixes.
"""

import config
from models.base import LLMClient


def get_client(model_name: str) -> LLMClient:
    """Return the correct LLMClient subclass for the given model name."""

    name = model_name.lower()

    # ── Google Gemini ─────────────────────────────────────────────────────────
    if model_name in config.GEMINI_MODELS:
        from models.gemini_client import GeminiClient
        return GeminiClient(model_name)

    # ── GitHub Models (free, Student Pack) — checked first ────────────────────
    if model_name in config.GITHUB_MODELS:
        from models.github_client import GitHubModelsClient
        return GitHubModelsClient(model_name)

    # ── Groq (cloud) — checked before Ollama; names overlap with Ollama ───────
    if model_name in config.GROQ_MODELS:
        from models.groq_client import GroqClient
        return GroqClient(model_name)

    # ── Ollama (local) ────────────────────────────────────────────────────────
    _ollama_prefixes = ("llama", "phi", "gemma", "mistral", "mixtral", "qwen", "deepseek", "orca")
    if any(name.startswith(p) for p in _ollama_prefixes):
        from models.ollama_client import OllamaClient
        return OllamaClient(model_name)

    # ── Anthropic Claude ──────────────────────────────────────────────────────
    if name.startswith("claude"):
        from models.claude_client import ClaudeClient
        return ClaudeClient(model_name)

    # ── OpenAI ────────────────────────────────────────────────────────────────
    if name.startswith("gpt") or name.startswith("o1") or name.startswith("o3"):
        from models.openai_client import OpenAIClient
        return OpenAIClient(model_name)

    raise ValueError(
        f"Unknown model '{model_name}'. "
        "Add a matching prefix in models/factory.py or pass a supported name."
    )
