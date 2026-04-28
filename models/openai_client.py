"""
OpenAI GPT client.

Requires:
  - pip install openai
  - OPENAI_API_KEY set in .env
"""

import time
import openai
from models.base import LLMClient
import config


class OpenAIClient(LLMClient):

    def __init__(self, model_name: str = config.OPENAI_MODEL):
        self._model_name = model_name
        self._client: openai.OpenAI | None = None

    @property
    def model_id(self) -> str:
        return self._model_name   # e.g. "gpt-4o-mini"

    def _get_client(self) -> openai.OpenAI:
        if self._client is None:
            self._client = openai.OpenAI(api_key=config.OPENAI_API_KEY)
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
                response = client.chat.completions.create(
                    model=self._model_name,
                    max_tokens=config.MAX_TOKENS,
                    temperature=config.TEMPERATURE,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user",   "content": user_prompt},
                    ],
                )
                return response.choices[0].message.content or ""

            except openai.RateLimitError:
                wait = backoff_base ** attempt
                print(f"  [rate limit] waiting {wait:.0f}s…")
                time.sleep(wait)

            except openai.APIStatusError as e:
                if attempt == retries - 1:
                    raise
                wait = backoff_base ** attempt
                print(f"  [API error {e.status_code}] waiting {wait:.0f}s…")
                time.sleep(wait)

        return ""
