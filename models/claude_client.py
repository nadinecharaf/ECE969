"""
Anthropic Claude client.

Requires:
  - pip install anthropic
  - ANTHROPIC_API_KEY set in .env
"""

import time
import anthropic
from models.base import LLMClient
import config


class ClaudeClient(LLMClient):

    def __init__(self, model_name: str = config.CLAUDE_MODEL):
        self._model_name = model_name
        self._client: anthropic.Anthropic | None = None

    @property
    def model_id(self) -> str:
        # "claude-haiku-4-5-20251001" → "claude-haiku"
        return "claude-" + self._model_name.split("-")[1]

    def _get_client(self) -> anthropic.Anthropic:
        if self._client is None:
            self._client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
        return self._client

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        retries: int = 3,
        backoff_base: float = 2.0,
    ) -> str:
        client = self._get_client()

        for attempt in range(retries):
            try:
                message = client.messages.create(
                    model=self._model_name,
                    max_tokens=config.MAX_TOKENS,
                    temperature=config.TEMPERATURE,
                    system=system_prompt,
                    messages=[{"role": "user", "content": user_prompt}],
                )
                return message.content[0].text

            except anthropic.RateLimitError:
                wait = backoff_base ** attempt
                print(f"  [rate limit] waiting {wait:.0f}s…")
                time.sleep(wait)

            except anthropic.APIStatusError as e:
                if attempt == retries - 1:
                    raise
                wait = backoff_base ** attempt
                print(f"  [API error {e.status_code}] waiting {wait:.0f}s…")
                time.sleep(wait)

        return ""
