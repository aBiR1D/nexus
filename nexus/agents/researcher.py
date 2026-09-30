"""Built-in research/analysis agent."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from nexus.agent import Agent
from nexus.context import Context


class ResearchOutput(BaseModel):
    answer: str = Field(description="The main answer or finding")
    confidence: float = Field(ge=0, le=1, description="Confidence in the answer (0-1)")
    sources: list[str] = Field(default_factory=list, description="Key sources or reasoning steps")
    follow_up_questions: list[str] = Field(default_factory=list)


class ResearcherAgent(Agent):
    """Researches a topic and provides structured findings.

    Usage:
        researcher = ResearcherAgent()
        result = await researcher(
            question="What's the best way to handle database migrations in Python?",
            ctx=ctx,
        )
        print(result.answer)
    """

    def __init__(self, **kwargs: Any):
        super().__init__(
            name="researcher",
            description="Researches topics and provides structured findings",
            tags=["research", "analysis"],
            **kwargs,
        )

    async def run(self, question: str, ctx: Context, **kwargs: Any) -> ResearchOutput:
        prior_research = await ctx.memory.recall(ctx.agent_name, question)
        context_hint = ""
        if prior_research:
            items = [str(p.value) if hasattr(p, 'value') else str(p) for p in prior_research[:3]]
            context_hint = f"\n\nRelevant prior findings:\n" + "\n".join(items)

        result = await ctx.llm.generate_structured(
            prompt=(
                f"Research this question thoroughly:\n{question}\n"
                f"Provide a clear answer, confidence level, reasoning sources, "
                f"and follow-up questions.{context_hint}"
            ),
            output_type=ResearchOutput,
            system=(
                "You are an expert researcher. Provide thorough, well-reasoned answers. "
                "Be honest about uncertainty. Cite your reasoning."
            ),
        )

        await ctx.memory.remember(
            ctx.agent_name,
            question[:100],
            result.answer[:500],
        )

        return result
