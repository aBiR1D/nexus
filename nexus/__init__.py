"""
Nexus — The Agent Runtime

Build reliable, composable, observable AI agents with persistent memory and guardrails.

    import nexus

    @nexus.agent(name="greeter")
    async def greet(name: str, ctx: nexus.Context) -> str:
        return await ctx.llm.generate(f"Write a greeting for {name}")

    result = await nexus.run(greet, name="World")
"""

__version__ = "0.1.0"

from nexus.agent import Agent, agent
from nexus.context import Context
from nexus.runtime import Runtime, run, run_sync
from nexus.types import AgentSpec, Severity, SpanStatus

from nexus.compose import Graph, Parallel, Pipeline, Router
from nexus.guardrails import Guardrail, guardrail
from nexus.observe import Span, Trace
from nexus.testing import AgentTest, mock_llm

__all__ = [
    "Agent",
    "AgentSpec",
    "AgentTest",
    "Context",
    "Graph",
    "Guardrail",
    "Parallel",
    "Pipeline",
    "Router",
    "Runtime",
    "Severity",
    "Span",
    "SpanStatus",
    "Trace",
    "agent",
    "guardrail",
    "mock_llm",
    "run",
    "run_sync",
]
