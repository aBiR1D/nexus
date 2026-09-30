"""Execution context — everything an agent needs to do its job."""

from __future__ import annotations

from typing import Any

from nexus.guardrails.engine import GuardrailEngine
from nexus.llm.provider import LLMProvider
from nexus.llm.mock import MockLLM
from nexus.memory.base import MemoryStore
from nexus.memory.local import LocalMemory
from nexus.observe.trace import Trace


class Context:
    """The execution context injected into every agent.

    Provides access to:
        - ctx.llm       — LLM provider (generate, classify, evaluate)
        - ctx.memory    — Persistent memory (remember, recall)
        - ctx.trace     — Execution tracing
        - ctx.guardrails — Guardrail engine
        - ctx.state     — Shared mutable state for pipeline steps
        - ctx.metadata  — Read-only metadata about the current execution
    """

    def __init__(
        self,
        llm: LLMProvider | MockLLM | None = None,
        memory: MemoryStore | None = None,
        trace: Trace | None = None,
        guardrails: GuardrailEngine | None = None,
        state: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        agent_name: str = "",
    ):
        self.llm: LLMProvider | MockLLM = llm or LLMProvider()
        self.memory: MemoryStore = memory or LocalMemory()
        self.trace: Trace = trace or Trace()
        self.guardrails: GuardrailEngine = guardrails or GuardrailEngine()
        self.state: dict[str, Any] = state or {}
        self.metadata: dict[str, Any] = metadata or {}
        self.agent_name: str = agent_name

    def child(self, agent_name: str = "", **extra_state: Any) -> Context:
        """Create a child context for sub-agent calls (shared memory, trace, and state)."""
        merged_state = {**self.state, **extra_state}
        return Context(
            llm=self.llm,
            memory=self.memory,
            trace=self.trace,
            guardrails=self.guardrails,
            state=merged_state,
            metadata=self.metadata,
            agent_name=agent_name or self.agent_name,
        )
