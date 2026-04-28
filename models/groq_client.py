"""
Groq client — fast cloud inference via the Groq API (OpenAI-compatible).

Requirements:
  - pip install groq
  - GROQ_API_KEY set in .env  (get key at https://console.groq.com)

Available models (as of 2026):
  llama-3.3-70b-versatile   — strongest, recommended
  llama-3.1-8b-instant      — fast, matches local llama3.1:8b
  mixtral-8x7b-32768        — Mixtral MoE, 32k context
"""

import time
from groq import Groq, APIError, RateLimitError
from models.base import LLMClient
import config


class GroqClient(LLMClient):

    def __init__(self, model_name: str = "llama-3.3-70b-versatile"):
        if not config.GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY is not set. Add it to your .env file.")
        self._model_name = model_name
        self._client = Groq(api_key=config.GROQ_API_KEY)

    @property
    def model_id(self) -> str:
        # "llama-3.3-70b-versatile" → safe for filenames as-is
        return self._model_name

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        retries: int = 4,
        backoff_base: float = 2.0,
    ) -> str:
        """Call the Groq API and return the response text."""

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
                content = response.choices[0].message.content or ""
                time.sleep(config.GROQ_INTER_CALL_DELAY)
                return content

            except RateLimitError:
                wait = min(60.0, backoff_base ** (attempt + 1))
                print(f"  [Groq rate limit] retrying in {wait:.0f}s…")
                time.sleep(wait)

            except APIError as e:
                if attempt == retries - 1:
                    raise
                wait = backoff_base ** attempt
                print(f"  [Groq API error] {e} — retrying in {wait:.0f}s…")
                time.sleep(wait)

        return ""