"""
GitHub Models client — free API access via GitHub Student Developer Pack.

Uses the OpenAI-compatible endpoint at models.inference.ai.azure.com.
Authentication: GitHub Personal Access Token (ghp_...).

Available models (as of 2026):
  gpt-4o                  — OpenAI GPT-4o
  gpt-4o-mini             — OpenAI GPT-4o-mini (fastest)
  Llama-3.3-70B-Instruct  — Meta Llama 3.3 70B
  Phi-4                   — Microsoft Phi-4

Rate limits (free tier):
  Low-tier models  — 15 req/min, 150 req/day
  High-tier models — 10 req/min,  50 req/day
"""

import time
from openai import OpenAI, RateLimitError, APIStatusError
from models.base import LLMClient
import config


class GitHubModelsClient(LLMClient):

    def __init__(self, model_name: str = "gpt-4o-mini"):
        if not config.GITHUB_TOKEN:
            raise ValueError("GITHUB_TOKEN is not set. Add it to your .env file.")
        self._model_name = model_name
        self._client = OpenAI(
            base_url=config.GITHUB_MODELS_BASE_URL,
            api_key=config.GITHUB_TOKEN,
        )

    @property
    def model_id(self) -> str:
        # "gpt-4o-mini" → "gpt-4o-mini"  (safe for filenames as-is)
        return self._model_name.lower().replace(" ", "-")

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        retries: int = 4,
        backoff_base: float = 3.0,
    ) -> str:
        """Call the GitHub Models API and return the response text."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ]

        for attempt in range(retries):
            try:
                response = self._client.chat.completions.create(
                    model=self._model_name,
                    messages=messages,
                    temperature=config.TEMPERATURE,
                    max_tokens=config.MAX_TOKENS,
                )
                return response.choices[0].message.content or ""

            except RateLimitError:
                wait = backoff_base ** (attempt + 1)
                print(f"  [GitHub rate limit] retrying in {wait:.0f}s…")
                time.sleep(wait)

            except APIStatusError as e:
                if attempt == retries - 1:
                    raise
                wait = backoff_base ** attempt
                print(f"  [GitHub API error {e.status_code}] {e.message} — retrying in {wait:.0f}s…")
                time.sleep(wait)

        return ""
