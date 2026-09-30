"""Conditional router — dispatches to different agents based on a routing function."""

from __future__ import annotations

from typing import Any, Callable

from nexus.agent import Agent
from nexus.context import Context


class Router(Agent):
    """Routes to different agents based on a condition.

    Usage:
        router = Router(
            routes={
                "code": code_agent,
                "review": review_agent,
                "docs": docs_agent,
            },
            route_fn=lambda input, ctx: ctx.state.get("task_type", "code"),
            name="task_router",
        )
        result = await router(task="fix the bug", ctx=ctx)

    Or with LLM-based routing:
        router = Router(
            routes={"code": coder, "review": reviewer, "docs": writer},
            route_fn="auto",  # Uses LLM to classify
            name="smart_router",
        )
    """

    def __init__(
        self,
        routes: dict[str, Agent],
        route_fn: Callable[[Any, Context], str] | str = "auto",
        name: str = "router",
        description: str = "",
        default_route: str = "",
    ):
        super().__init__(name=name, description=description or f"Router with {len(routes)} routes")
        self.routes = routes
        self._route_fn = route_fn
        self.default_route = default_route or next(iter(routes))

    async def run(self, ctx: Context, **kwargs: Any) -> Any:
        """Route to the appropriate agent."""
        with ctx.trace.span(f"router:{self.name}", agent_name=self.name) as span:
            # Determine route
            if self._route_fn == "auto":
                categories = list(self.routes.keys())
                input_text = str(kwargs)
                route_key = await ctx.llm.classify(input_text, categories)
                span.add_event("auto_route", {"input": input_text[:200], "route": route_key})
            elif callable(self._route_fn):
                route_key = self._route_fn(kwargs, ctx)
            else:
                route_key = str(self._route_fn)

            if route_key not in self.routes:
                route_key = self.default_route
                span.add_event("fallback_route", {"route": route_key})

            target = self.routes[route_key]
            span.add_event("dispatching", {"route": route_key, "agent": target.name})

            result = await target(ctx=ctx.child(agent_name=target.name), **kwargs)
            span.succeed()
            return result
