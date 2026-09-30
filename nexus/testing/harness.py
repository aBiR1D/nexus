"""Agent test harness — test AI agents deterministically."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Generator

from nexus.agent import Agent
from nexus.context import Context
from nexus.guardrails.engine import GuardrailEngine
from nexus.llm.mock import MockLLM
from nexus.memory.local import LocalMemory
from nexus.observe.trace import Trace
from nexus.runtime import Runtime
from nexus.types import AgentResult


class AgentTest:
    """Test harness for running agents with mocked LLMs and assertions.

    Usage:
        test = AgentTest(mock_responses=["Hello!"])
        result = await test.run(greet_agent, name="World")

        assert result.succeeded
        assert "Hello" in result.output
        test.mock.assert_called(times=1)
        test.mock.assert_prompt_contains("World")
    """

    def __init__(
        self,
        mock_responses: list[Any] | None = None,
        response_map: dict[str, Any] | None = None,
        memory_path: str = ":memory:",
    ):
        self.mock = MockLLM(
            responses=mock_responses,
            response_map=response_map,
        )
        self.memory = LocalMemory(memory_path) if memory_path != ":memory:" else _InMemoryStore()
        self.trace = Trace()
        self.runtime = Runtime(
            llm=self.mock,
            memory=self.memory,
        )
        self.runtime.trace = self.trace

    async def run(self, agent: Agent, **kwargs: Any) -> AgentResult:
        """Run an agent with mocked LLM and return the result."""
        return await self.runtime.run(agent, **kwargs)

    def assert_output_contains(self, text: str) -> None:
        result = self.runtime.last_result
        assert result is not None, "No result — agent hasn't been run yet"
        assert text in str(result.output), f"Expected '{text}' in output: {result.output}"

    def assert_succeeded(self) -> None:
        result = self.runtime.last_result
        assert result is not None, "No result"
        assert result.succeeded, f"Agent failed: {result.error}"

    def assert_failed(self) -> None:
        result = self.runtime.last_result
        assert result is not None, "No result"
        assert result.failed, f"Agent succeeded but expected failure"

    def reset(self) -> None:
        self.mock.reset()
        self.trace = Trace()
        self.runtime.trace = self.trace


class _InMemoryStore:
    """Minimal in-memory store for testing (no SQLite needed)."""

    def __init__(self):
        self._data: dict[str, list] = {}

    async def remember(self, agent_name, key, value, entry_type="semantic"):
        self._data.setdefault(agent_name, []).append({"key": key, "value": value})
        return "test-id"

    async def recall(self, agent_name, key, limit=5):
        entries = self._data.get(agent_name, [])
        return [e for e in entries if key.lower() in e["key"].lower()][:limit]

    async def recall_recent(self, agent_name, limit=10):
        return self._data.get(agent_name, [])[-limit:]

    async def forget(self, agent_name, key):
        entries = self._data.get(agent_name, [])
        before = len(entries)
        self._data[agent_name] = [e for e in entries if key.lower() not in e["key"].lower()]
        return before - len(self._data[agent_name])

    async def count(self, agent_name):
        return len(self._data.get(agent_name, []))

    async def clear(self, agent_name):
        self._data[agent_name] = []


@contextmanager
def mock_llm(
    responses: list[Any] | None = None,
    response_map: dict[str, Any] | None = None,
) -> Generator[MockLLM, None, None]:
    """Context manager for quick LLM mocking.

    with mock_llm(responses=["Hello"]) as mock:
        ctx = Context(llm=mock)
        result = await my_agent(ctx=ctx, name="World")
        mock.assert_called(times=1)
    """
    yield MockLLM(responses=responses, response_map=response_map)
