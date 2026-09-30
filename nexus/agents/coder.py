"""Built-in code generation agent."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from nexus.agent import Agent
from nexus.context import Context
from nexus.guardrails.rules import no_code_execution


class CodeOutput(BaseModel):
    """Structured output from the coder agent."""
    code: str = Field(description="The generated code")
    language: str = Field(description="Programming language")
    explanation: str = Field(description="Brief explanation of the code")
    files: list[dict[str, str]] = Field(
        default_factory=list,
        description="List of {path, content} for multi-file output",
    )


class CoderAgent(Agent):
    """Generates code from natural language descriptions.

    Usage:
        coder = CoderAgent()
        result = await coder(
            task="Write a Python function to sort a list of dicts by key",
            language="python",
            ctx=ctx,
        )
    """

    def __init__(self, **kwargs: Any):
        super().__init__(
            name="coder",
            description="Generates code from natural language descriptions",
            tags=["code", "generation"],
            **kwargs,
        )

    async def run(self, task: str, ctx: Context, language: str = "python", **kwargs: Any) -> CodeOutput:
        past_patterns = await ctx.memory.recall(ctx.agent_name, f"coding patterns {language}")
        context_hint = ""
        if past_patterns:
            patterns = [str(p.value) if hasattr(p, 'value') else str(p) for p in past_patterns[:3]]
            context_hint = f"\n\nLearned patterns from past tasks:\n" + "\n".join(patterns)

        result = await ctx.llm.generate_structured(
            prompt=(
                f"Write {language} code for this task:\n{task}\n"
                f"Include a clear explanation.{context_hint}"
            ),
            output_type=CodeOutput,
            system=(
                f"You are an expert {language} developer. Write clean, idiomatic, "
                f"well-documented code. Follow best practices."
            ),
        )

        await ctx.memory.remember(
            ctx.agent_name,
            f"coding patterns {language}",
            f"Task: {task[:100]} → Generated {len(result.code)} chars of {language}",
        )

        return result
