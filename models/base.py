"""
Abstract base class that every LLM client must implement.
Adding a new model = subclass this + register in factory.py.
"""

from abc import ABC, abstractmethod


class LLMClient(ABC):

    @abstractmethod
    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """Send a prompt and return the model's response as a string."""
        ...

    @property
    @abstractmethod
    def model_id(self) -> str:
        """Short identifier used in result filenames (e.g. 'llama3.2-3b')."""
        ...
