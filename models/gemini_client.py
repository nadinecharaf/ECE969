"""
Google Gemini client — free inference via Google AI Studio.

Requirements:
  - pip install google-genai
  - GEMINI_API_KEY_1 (and optionally GEMINI_API_KEY_2) set in .env
    Get keys at: https://aistudio.google.com → Get API Key

Free tier limits (per key):
  gemini-2.0-flash: 15 req/min, 1,500 req/day
  gemini-1.5-flash: 15 req/min, 1,500 req/day

With 2 keys: 3,000 req/day — enough for a full 1,875-call study in one day.
Key rotation: when key 1 hits its daily limit, automatically switches to key 2.
"""

import time
from google import genai
from google.genai import errors as genai_errors
from models.base import LLMClient
import config


class GeminiClient(LLMClient):

    def __init__(self, model_name: str = "gemini-2.0-flash"):
        if not config.GEMINI_API_KEYS:
            raise ValueError(
                "No Gemini API keys found. "
                "Add GEMINI_API_KEY_1 (and optionally GEMINI_API_KEY_2) to your .env file."
            )
        self._model_name  = model_name
        self._keys        = list(config.GEMINI_API_KEYS)
        self._key_index   = 0
        self._client      = genai.Client(api_key=self._keys[0])

    def _rotate_key(self) -> bool:
        """Switch to the next available API key. Returns False if no more keys."""
        self._key_index += 1
        if self._key_index >= len(self._keys):
            return False
        print(f"  [Gemini] switching to API key {self._key_index + 1}…")
        self._client = genai.Client(api_key=self._keys[self._key_index])
        return True

    @property
    def model_id(self) -> str:
        return self._model_name   # e.g. "gemini-2.0-flash"

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        retries: int = 4,
        backoff_base: float = 2.0,
    ) -> str:
        """Call the Gemini API and return the response text."""

        for attempt in range(retries):
            try:
                response = self._client.models.generate_content(
                    model=self._model_name,
                    contents=user_prompt,
                    config=genai.types.GenerateContentConfig(
                        system_instruction=system_prompt,
                        temperature=config.TEMPERATURE,
                        max_output_tokens=config.MAX_TOKENS,
                    ),
                )
                content = response.text or ""
                time.sleep(config.GEMINI_INTER_CALL_DELAY)
                return content

            except Exception as e:
                err_str = str(e).lower()
                print(f"  [Gemini raw error] {e}")   # always print full error for diagnosis
                is_rate_limit = "429" in str(e) or "quota" in err_str or "exhausted" in err_str or "rate" in err_str
                is_daily      = "per day" in err_str or "daily" in err_str or "requests per day" in err_str

                if is_rate_limit:
                    if is_daily:
                        if self._rotate_key():
                            continue
                    wait = min(60.0, backoff_base ** (attempt + 1))
                    print(f"  [Gemini rate limit] retrying in {wait:.0f}s…")
                    time.sleep(wait)
                else:
                    if attempt == retries - 1:
                        raise
                    wait = backoff_base ** attempt
                    print(f"  [Gemini error] retrying in {wait:.0f}s…")
                    time.sleep(wait)

        return ""
