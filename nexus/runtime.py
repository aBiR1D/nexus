"""Agent runtime — the execution engine that ties everything together."""

from __future__ import annotations

import asyncio
from typing import Any

from nexus.agent import Agent
from nexus.config import get_settings
from nexus.context import Context
from nexus.guardrails.engine import GuardrailEngine
from nexus.guardrails.rules import Guardrail
from nexus.llm.mock import MockLLM
from nexus.llm.provider import LLMProvider
from nexus.memory.local import LocalMemory
from nexus.observe.trace import Trace
from nexus.types import AgentResult, SpanStatus


class Runtime:
    """The Nexus agent runtime — manages execution, memory, tracing, and guardrails.

    Usage:
        runtime = Runtime()
        result = await runtime.run(my_agent, task="Fix the bug")
        print(result.output)
        print(runtime.trace.print_tree())
    """

    def __init__(
        self,
        llm: LLMProvider | MockLLM | None = None,
        memory: LocalMemory | None = None,
        guardrails: list[Guardrail] | None = None,
    ):
        settings = get_settings()
        self.llm = llm or LLMProvider()
        self.memory = memory or LocalMemory(settings.memory_path)
        self.guardrail_engine = GuardrailEngine(guardrails or [])
        self.trace = Trace()
        self._results: list[AgentResult] = []

    def create_context(self, agent_name: str = "", **state: Any) -> Context:
        """Create an execution context for an agent."""
        return Context(
            llm=self.llm,
            memory=self.memory,
            trace=self.trace,
            guardrails=self.guardrail_engine,
            state=state,
            agent_name=agent_name,
        )

    async def run(self, agent: Agent, **kwargs: Any) -> AgentResult:
        """Execute an agent and return a structured result."""
        ctx = self.create_context(agent_name=agent.name, **kwargs)
        kwargs["ctx"] = ctx

        result = AgentResult(
            agent_name=agent.name,
            trace_id=self.trace.trace_id,
        )

        try:
            output = await agent(**kwargs)
            result.output = output
            result.status = SpanStatus.SUCCESS
        except Exception as e:
            result.error = str(e)
            result.status = SpanStatus.FAILED

        result.duration_ms = self.trace.duration_ms
        result.tokens_used = getattr(self.llm, "total_tokens", 0)
        result.guardrails_checked = self.guardrail_engine.checks_run
        result.guardrails_passed = self.guardrail_engine.checks_passed

        self._results.append(result)
        return result

    @property
    def last_result(self) -> AgentResult | None:
        return self._results[-1] if self._results else None

    @property
    def results(self) -> list[AgentResult]:
        return list(self._results)


async def run(agent: Agent, **kwargs: Any) -> Any:
    """Quick-run an agent with a fresh runtime. Returns the raw output."""
    runtime = Runtime()
    result = await runtime.run(agent, **kwargs)
    if result.failed:
        raise RuntimeError(f"Agent '{agent.name}' failed: {result.error}")
    return result.output


def run_sync(agent: Agent, **kwargs: Any) -> Any:
    """Synchronous wrapper for running an agent."""
    return asyncio.run(run(agent, **kwargs))
