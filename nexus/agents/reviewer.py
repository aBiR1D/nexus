"""Built-in code review agent."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from nexus.agent import Agent
from nexus.context import Context
from nexus.types import Severity


class ReviewIssue(BaseModel):
    line: int = 0
    severity: str = "medium"
    category: str = ""
    message: str = ""
    suggestion: str = ""


class ReviewOutput(BaseModel):
    score: int = Field(ge=1, le=10, description="Overall quality score 1-10")
    summary: str = Field(description="Brief review summary")
    issues: list[ReviewIssue] = Field(default_factory=list)
    strengths: list[str] = Field(default_factory=list)
    approved: bool = Field(description="Whether the code passes review")


class ReviewerAgent(Agent):
    """Reviews code for bugs, style issues, and best practice violations.

    Usage:
        reviewer = ReviewerAgent()
        result = await reviewer(
            code="def foo(x): return x+1",
            language="python",
            ctx=ctx,
        )
        if not result.approved:
            for issue in result.issues:
                print(f"Line {issue.line}: {issue.message}")
    """

    def __init__(self, min_score: int = 7, **kwargs: Any):
        super().__init__(
            name="reviewer",
            description="Reviews code for bugs, style, and best practices",
            tags=["code", "review", "quality"],
            **kwargs,
        )
        self.min_score = min_score

    async def run(self, code: str, ctx: Context, language: str = "python", **kwargs: Any) -> ReviewOutput:
        past_reviews = await ctx.memory.recall(ctx.agent_name, f"review patterns {language}")
        history_hint = ""
        if past_reviews:
            items = [str(p.value) if hasattr(p, 'value') else str(p) for p in past_reviews[:3]]
            history_hint = "\n\nCommon issues from past reviews:\n" + "\n".join(items)

        result = await ctx.llm.generate_structured(
            prompt=(
                f"Review this {language} code:\n```{language}\n{code}\n```\n"
                f"Score from 1-10. List specific issues with line numbers.\n"
                f"Set approved=true if score >= {self.min_score}.{history_hint}"
            ),
            output_type=ReviewOutput,
            system=(
                f"You are a senior {language} code reviewer. Be thorough but fair. "
                f"Focus on bugs, security, performance, and readability."
            ),
        )

        if result.issues:
            categories = [i.category for i in result.issues if i.category]
            if categories:
                await ctx.memory.remember(
                    ctx.agent_name,
                    f"review patterns {language}",
                    f"Common issues: {', '.join(set(categories))}",
                )

        return result
