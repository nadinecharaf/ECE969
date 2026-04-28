"""
Ollama client — talks to the local Ollama server (http://localhost:11434).

Requirements:
  - Ollama installed and running  (https://ollama.com)
  - Model pulled:  ollama pull llama3.2:3b
                   ollama pull phi4-mini
"""

import time
import ollama
from models.base import LLMClient
import config


class OllamaClient(LLMClient):

    def __init__(self, model_name: str = config.OLLAMA_MODEL):
        self._model_name = model_name

    @property
    def model_id(self) -> str:
        # "llama3.2:3b" → "llama3.2-3b"  (safe for filenames)
        return self._model_name.replace(":", "-")

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        retries: int = 3,
        backoff_base: float = 2.0,
    ) -> str:
        """Call the local Ollama server and return the response text."""

        messages = [
            {"role": "system",  "content": system_prompt},
            {"role": "user",    "content": user_prompt},
        ]

        for attempt in range(retries):
            try:
                response = ollama.chat(
                    model=self._model_name,
                    messages=messages,
                    options={
                        "temperature": config.TEMPERATURE,
                        "num_predict": config.MAX_TOKENS,
                    },
                )
                return response["message"]["content"]

            except Exception as e:
                if attempt == retries - 1:
                    raise
                wait = backoff_base ** attempt
                print(f"  [Ollama error] {e} — retrying in {wait:.0f}s…")
                time.sleep(wait)

        return ""
