"""Agent definition — the @agent decorator and Agent base class."""

from __future__ import annotations

import asyncio
import inspect
import time
from functools import wraps
from typing import Any, Callable, Coroutine

from nexus.context import Context
from nexus.guardrails.rules import Guardrail
from nexus.types import AgentResult, AgentSpec, RetryPolicy, SpanStatus


class Agent:
    """A wrapped callable that executes with context, memory, guardrails, and tracing.

    Created via the @agent decorator:

        @nexus.agent(name="greeter")
        async def greet(name: str, ctx: nexus.Context) -> str:
            return await ctx.llm.generate(f"Greet {name}")

    Or by subclassing:

        class MyAgent(nexus.Agent):
            async def run(self, task: str, ctx: nexus.Context) -> str:
                return await ctx.llm.generate(task)
    """

    def __init__(
        self,
        fn: Callable | None = None,
        name: str = "",
        description: str = "",
        guardrails: list[Guardrail] | None = None,
        retry: RetryPolicy | None = None,
        timeout: float = 120.0,
        tags: list[str] | None = None,
    ):
        self._fn = fn
        self.spec = AgentSpec(
            name=name or (fn.__name__ if fn else self.__class__.__name__),
            description=description or (fn.__doc__ or "" if fn else self.__class__.__doc__ or ""),
            guardrails=[g.name for g in (guardrails or [])],
            retry_policy=retry or RetryPolicy(),
            timeout=timeout,
            tags=tags or [],
        )
        self._guardrails = guardrails or []

    @property
    def name(self) -> str:
        return self.spec.name

    async def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Execute the agent — injects context if not provided."""
        ctx = kwargs.get("ctx")
        if ctx is None:
            for arg in args:
                if isinstance(arg, Context):
                    ctx = arg
                    break

        if ctx is None:
            ctx = Context(agent_name=self.name)
            kwargs["ctx"] = ctx

        ctx.agent_name = self.name

        return await self._execute_with_trace(ctx, args, kwargs)

    async def _execute_with_trace(
        self, ctx: Context, args: tuple, kwargs: dict
    ) -> Any:
        """Execute with full tracing, guardrails, and retry."""
        with ctx.trace.span(self.name, agent_name=self.name) as span:
            input_data = {
                k: v for k, v in kwargs.items()
                if k != "ctx" and not isinstance(v, Context)
            }
            span.input_data = input_data
            span.add_event("agent_start", {"input": str(input_data)[:500]})

            # Input guardrails
            for g in self._guardrails:
                if g.applies_to in ("input", "both"):
                    ctx.guardrails.add(g)
            try:
                ctx.guardrails.check_input(input_data)
            except Exception as e:
                span.status = SpanStatus.GUARDRAIL_BLOCKED
                span.fail(str(e))
                raise

            # Execute with retries
            retry = self.spec.retry_policy or RetryPolicy()
            last_error: Exception | None = None

            for attempt in range(retry.max_retries + 1):
                try:
                    span.add_event("attempt", {"attempt": attempt + 1})

                    if self._fn:
                        result = self._fn(*args, **kwargs)
                    else:
                        filtered_kwargs = {
                            k: v for k, v in kwargs.items()
                            if k != "ctx"
                        }
                        result = self.run(**filtered_kwargs, ctx=ctx)

                    if asyncio.iscoroutine(result):
                        result = await result

                    # Output guardrails
                    for g in self._guardrails:
                        if g.applies_to in ("output", "both"):
                            ctx.guardrails.add(g)
                    try:
                        ctx.guardrails.check_output(result)
                    except Exception as e:
                        span.status = SpanStatus.GUARDRAIL_BLOCKED
                        span.fail(str(e))
                        raise

                    span.succeed(str(result)[:1000] if result else None)
                    span.add_event("agent_end", {"output_preview": str(result)[:200]})
                    return result

                except Exception as e:
                    last_error = e
                    if attempt < retry.max_retries:
                        wait = retry.backoff_factor * (2 ** attempt)
                        span.add_event("retry", {"attempt": attempt + 1, "wait": wait, "error": str(e)})
                        await asyncio.sleep(wait)
                    else:
                        span.fail(str(e))
                        raise

    async def run(self, *args: Any, **kwargs: Any) -> Any:
        """Override this method when subclassing Agent."""
        raise NotImplementedError(
            f"Agent '{self.name}' must implement run() or be created with @agent decorator"
        )

    def __repr__(self) -> str:
        return f"Agent({self.name!r})"


def agent(
    name: str = "",
    description: str = "",
    guardrails: list[Guardrail] | None = None,
    retry: RetryPolicy | None = None,
    timeout: float = 120.0,
    tags: list[str] | None = None,
) -> Callable:
    """Decorator to turn a function into a Nexus agent.

    @nexus.agent(name="summarizer", tags=["nlp"])
    async def summarize(text: str, ctx: nexus.Context) -> str:
        return await ctx.llm.generate(f"Summarize: {text}")
    """
    def decorator(fn: Callable) -> Agent:
        return Agent(
            fn=fn,
            name=name or fn.__name__,
            description=description or fn.__doc__ or "",
            guardrails=guardrails,
            retry=retry,
            timeout=timeout,
            tags=tags,
        )
    return decorator
