"""Sequential pipeline — agents run one after another, output flows forward."""

from __future__ import annotations

from typing import Any

from nexus.agent import Agent
from nexus.context import Context
from nexus.types import SpanStatus


class Pipeline(Agent):
    """Runs agents sequentially, passing each output as input to the next.

    Usage:
        pipeline = Pipeline(
            steps=[agent_a, agent_b, agent_c],
            name="my_pipeline",
        )
        result = await pipeline(initial_input="hello", ctx=ctx)

    Each step receives:
        - The previous step's output in ctx.state["pipeline_input"]
        - All accumulated state in ctx.state
    """

    def __init__(
        self,
        steps: list[Agent],
        name: str = "pipeline",
        description: str = "",
        pass_output_as: str = "pipeline_input",
    ):
        super().__init__(name=name, description=description or f"Pipeline of {len(steps)} agents")
        self.steps = steps
        self.pass_output_as = pass_output_as

    async def run(self, ctx: Context, **kwargs: Any) -> Any:
        """Execute all steps sequentially."""
        current_input = kwargs

        with ctx.trace.span(f"pipeline:{self.name}", agent_name=self.name) as pipeline_span:
            for i, step in enumerate(self.steps):
                step_ctx = ctx.child(
                    agent_name=step.name,
                    **{self.pass_output_as: current_input},
                )

                with ctx.trace.span(
                    f"step[{i}]:{step.name}", agent_name=step.name
                ) as step_span:
                    try:
                        if isinstance(current_input, dict):
                            result = await step(ctx=step_ctx, **current_input)
                        else:
                            result = await step(ctx=step_ctx, **{self.pass_output_as: current_input})

                        current_input = result
                        step_span.succeed(str(result)[:500])
                    except Exception as e:
                        step_span.fail(str(e))
                        pipeline_span.fail(f"Step {i} ({step.name}) failed: {e}")
                        raise

            pipeline_span.succeed()
            return current_input
