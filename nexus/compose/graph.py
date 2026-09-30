"""DAG-based workflow — directed acyclic graph of agents with dependencies."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

from nexus.agent import Agent
from nexus.context import Context


@dataclass
class Node:
    """A node in the agent graph."""
    agent: Agent
    depends_on: list[str] = field(default_factory=list)


class Graph(Agent):
    """Execute agents as a DAG — respecting dependencies, parallelizing where possible.

    Usage:
        graph = Graph(name="build_pipeline")
        graph.add_node("parse", parser_agent)
        graph.add_node("lint", linter_agent, depends_on=["parse"])
        graph.add_node("test", test_agent, depends_on=["parse"])
        graph.add_node("deploy", deployer, depends_on=["lint", "test"])

        results = await graph(code="...", ctx=ctx)
    """

    def __init__(self, name: str = "graph", description: str = ""):
        super().__init__(name=name, description=description)
        self.nodes: dict[str, Node] = {}

    def add_node(
        self, name: str, agent: Agent, depends_on: list[str] | None = None
    ) -> Graph:
        self.nodes[name] = Node(agent=agent, depends_on=depends_on or [])
        return self

    async def run(self, ctx: Context, **kwargs: Any) -> dict[str, Any]:
        """Execute the DAG respecting dependencies."""
        with ctx.trace.span(f"graph:{self.name}", agent_name=self.name) as span:
            completed: dict[str, Any] = {}
            pending = set(self.nodes.keys())

            while pending:
                ready = [
                    name for name in pending
                    if all(dep in completed for dep in self.nodes[name].depends_on)
                ]

                if not ready:
                    remaining = ", ".join(pending)
                    span.fail(f"Deadlock: no ready nodes. Remaining: {remaining}")
                    raise RuntimeError(f"Graph deadlock: {remaining}")

                tasks = []
                for name in ready:
                    node = self.nodes[name]
                    child_ctx = ctx.child(
                        agent_name=node.agent.name,
                        graph_results=completed,
                    )
                    merged_kwargs = {**kwargs, "graph_results": completed}
                    tasks.append((name, self._run_node(node.agent, child_ctx, merged_kwargs)))

                results = await asyncio.gather(
                    *[t[1] for t in tasks], return_exceptions=True
                )

                for (name, _), result in zip(tasks, results):
                    if isinstance(result, Exception):
                        completed[name] = {"error": str(result)}
                        span.add_event("node_failed", {"node": name, "error": str(result)})
                    else:
                        completed[name] = result
                    pending.discard(name)

            span.succeed()
            return completed

    @staticmethod
    async def _run_node(agent: Agent, ctx: Context, kwargs: dict) -> Any:
        return await agent(ctx=ctx, **kwargs)

    def validate(self) -> list[str]:
        """Validate the graph has no cycles or missing dependencies."""
        errors = []
        for name, node in self.nodes.items():
            for dep in node.depends_on:
                if dep not in self.nodes:
                    errors.append(f"Node '{name}' depends on '{dep}' which doesn't exist")

        visited: set[str] = set()
        path: set[str] = set()

        def has_cycle(name: str) -> bool:
            if name in path:
                return True
            if name in visited:
                return False
            visited.add(name)
            path.add(name)
            for dep in self.nodes.get(name, Node(agent=Agent())).depends_on:
                if has_cycle(dep):
                    return True
            path.discard(name)
            return False

        for name in self.nodes:
            if has_cycle(name):
                errors.append(f"Cycle detected involving node '{name}'")
                break

        return errors
