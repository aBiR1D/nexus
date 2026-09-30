"""Parallel execution — run multiple agents concurrently and collect results."""

from __future__ import annotations

import asyncio
from typing import Any

from nexus.agent import Agent
from nexus.context import Context


class Parallel(Agent):
    """Run multiple agents concurrently and collect all results.

    Usage:
        parallel = Parallel(
            agents=[analyzer, reviewer, tester],
            name="parallel_checks",
        )
        results = await parallel(code="def foo(): ...", ctx=ctx)
        # results == {"analyzer": ..., "reviewer": ..., "tester": ...}
    """

    def __init__(
        self,
        agents: list[Agent],
        name: str = "parallel",
        description: str = "",
        fail_fast: bool = False,
    ):
        super().__init__(name=name, description=description or f"Parallel({len(agents)} agents)")
        self.agents = agents
        self.fail_fast = fail_fast

    async def run(self, ctx: Context, **kwargs: Any) -> dict[str, Any]:
        """Execute all agents in parallel."""
        with ctx.trace.span(f"parallel:{self.name}", agent_name=self.name) as span:
            tasks = []
            for ag in self.agents:
                child_ctx = ctx.child(agent_name=ag.name)
                tasks.append(self._run_one(ag, child_ctx, kwargs))

            if self.fail_fast:
                results_list = await asyncio.gather(*tasks)
            else:
                results_list = await asyncio.gather(*tasks, return_exceptions=True)

            results: dict[str, Any] = {}
            for ag, result in zip(self.agents, results_list):
                if isinstance(result, Exception):
                    results[ag.name] = {"error": str(result)}
                    span.add_event("agent_failed", {"agent": ag.name, "error": str(result)})
                else:
                    results[ag.name] = result

            span.succeed()
            return results

    @staticmethod
    async def _run_one(ag: Agent, ctx: Context, kwargs: dict) -> Any:
        return await ag(ctx=ctx, **kwargs)
