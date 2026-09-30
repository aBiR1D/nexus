"""Mock LLM for deterministic agent testing."""

from __future__ import annotations

import json
from typing import Any, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class MockLLM:
    """Deterministic LLM mock for testing agents without real API calls.

    Usage:
        mock = MockLLM(responses=["Hello, World!"])
        result = await mock.generate("Say hello")
        assert result == "Hello, World!"

        # Structured output
        mock = MockLLM(responses=[ReviewResult(score=8, issues=[])])
        result = await mock.generate_structured("Review this", ReviewResult)
        assert result.score == 8

        # Conditional responses
        mock = MockLLM(response_map={
            "review": "LGTM",
            "fix": "Here's the fix...",
        })
    """

    def __init__(
        self,
        responses: list[Any] | None = None,
        response_map: dict[str, Any] | None = None,
        default_response: str = "Mock response",
    ):
        self._responses = list(responses or [])
        self._response_map = response_map or {}
        self._default = default_response
        self._call_index = 0
        self.calls: list[dict[str, Any]] = []
        self.total_tokens = 0
        self.call_count = 0

    def _record_call(self, prompt: str, **kwargs: Any) -> None:
        self.calls.append({"prompt": prompt, **kwargs})
        self.call_count += 1

    def _get_response(self, prompt: str) -> Any:
        for keyword, response in self._response_map.items():
            if keyword.lower() in prompt.lower():
                return response

        if self._call_index < len(self._responses):
            resp = self._responses[self._call_index]
            self._call_index += 1
            return resp

        return self._default

    async def generate(
        self,
        prompt: str,
        system: str = "",
        temperature: float = 0.2,
        max_tokens: int = 4096,
    ) -> str:
        self._record_call(prompt, system=system)
        response = self._get_response(prompt)
        if isinstance(response, BaseModel):
            return response.model_dump_json()
        return str(response)

    async def generate_structured(
        self,
        prompt: str,
        output_type: type[T],
        system: str = "",
        temperature: float = 0.1,
        max_tokens: int = 4096,
    ) -> T:
        self._record_call(prompt, system=system, output_type=output_type.__name__)
        response = self._get_response(prompt)
        if isinstance(response, output_type):
            return response
        if isinstance(response, BaseModel):
            return output_type.model_validate(response.model_dump())
        if isinstance(response, str):
            return output_type.model_validate_json(response)
        if isinstance(response, dict):
            return output_type.model_validate(response)
        raise TypeError(f"Cannot convert mock response to {output_type.__name__}: {response}")

    async def classify(self, text: str, categories: list[str], system: str = "") -> str:
        self._record_call(text, categories=categories)
        response = self._get_response(text)
        return str(response) if str(response) in categories else categories[0]

    async def evaluate(self, text: str, criteria: str, scale: tuple[int, int] = (1, 10)) -> int:
        self._record_call(text, criteria=criteria)
        response = self._get_response(text)
        try:
            return int(response)
        except (ValueError, TypeError):
            return (scale[0] + scale[1]) // 2

    def reset(self) -> None:
        self._call_index = 0
        self.calls.clear()
        self.call_count = 0

    def assert_called(self, times: int | None = None) -> None:
        if times is not None:
            assert self.call_count == times, f"Expected {times} calls, got {self.call_count}"
        else:
            assert self.call_count > 0, "Expected at least one call"

    def assert_prompt_contains(self, text: str, call_index: int = -1) -> None:
        assert self.calls, "No calls recorded"
        prompt = self.calls[call_index]["prompt"]
        assert text in prompt, f"Expected '{text}' in prompt, got: {prompt[:200]}"
