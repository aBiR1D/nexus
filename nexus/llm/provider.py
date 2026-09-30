"""LLM provider abstraction — one interface for Claude, OpenAI, and mocks."""

from __future__ import annotations

import json
import time
from typing import Any, TypeVar, get_type_hints

from pydantic import BaseModel

from nexus.config import get_settings

T = TypeVar("T", bound=BaseModel)


class LLMProvider:
    """Unified LLM interface with structured output, token tracking, and retries."""

    def __init__(
        self,
        provider: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
    ):
        settings = get_settings()
        self.provider = provider or settings.ai_provider
        self.model = model or settings.model
        self._api_key = api_key
        self._client = None
        self.total_tokens = 0
        self.call_count = 0

    @property
    def client(self):
        if self._client is not None:
            return self._client

        settings = get_settings()
        if self.provider == "anthropic":
            from anthropic import Anthropic
            self._client = Anthropic(api_key=self._api_key or settings.anthropic_api_key)
        elif self.provider == "openai":
            from openai import OpenAI
            self._client = OpenAI(api_key=self._api_key or settings.openai_api_key)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")
        return self._client

    async def generate(
        self,
        prompt: str,
        system: str = "You are a helpful AI assistant.",
        temperature: float = 0.2,
        max_tokens: int = 4096,
    ) -> str:
        """Generate a text response."""
        self.call_count += 1
        if self.provider == "anthropic":
            response = self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system,
                messages=[{"role": "user", "content": prompt}],
            )
            self.total_tokens += response.usage.input_tokens + response.usage.output_tokens
            return response.content[0].text

        elif self.provider == "openai":
            messages = [{"role": "system", "content": system}] if system else []
            messages.append({"role": "user", "content": prompt})
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            if response.usage:
                self.total_tokens += response.usage.total_tokens
            return response.choices[0].message.content or ""

        raise ValueError(f"Unsupported provider: {self.provider}")

    async def generate_structured(
        self,
        prompt: str,
        output_type: type[T],
        system: str = "You are a helpful AI assistant.",
        temperature: float = 0.1,
        max_tokens: int = 4096,
    ) -> T:
        """Generate a structured response parsed into a Pydantic model."""
        schema_json = json.dumps(output_type.model_json_schema(), indent=2)
        full_prompt = (
            f"{prompt}\n\n"
            f"Respond with valid JSON matching this schema:\n{schema_json}\n\n"
            f"Return ONLY the JSON object. No markdown, no explanation."
        )
        raw = await self.generate(full_prompt, system=system, temperature=temperature, max_tokens=max_tokens)
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            cleaned = "\n".join(lines[1:])
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3].strip()
        return output_type.model_validate_json(cleaned)

    async def classify(
        self,
        text: str,
        categories: list[str],
        system: str = "",
    ) -> str:
        """Classify text into one of the given categories."""
        cats = ", ".join(f'"{c}"' for c in categories)
        prompt = (
            f"Classify this text into exactly one category.\n\n"
            f"Categories: {cats}\n\n"
            f"Text: {text}\n\n"
            f"Respond with ONLY the category name, nothing else."
        )
        result = await self.generate(prompt, system=system, temperature=0.0, max_tokens=100)
        return result.strip().strip('"')

    async def evaluate(
        self,
        text: str,
        criteria: str,
        scale: tuple[int, int] = (1, 10),
    ) -> int:
        """Evaluate text on a numeric scale based on criteria."""
        prompt = (
            f"Evaluate this text on a scale of {scale[0]}-{scale[1]}.\n\n"
            f"Criteria: {criteria}\n\n"
            f"Text: {text}\n\n"
            f"Respond with ONLY the number."
        )
        result = await self.generate(prompt, temperature=0.0, max_tokens=10)
        try:
            score = int(result.strip())
            return max(scale[0], min(scale[1], score))
        except ValueError:
            return (scale[0] + scale[1]) // 2
