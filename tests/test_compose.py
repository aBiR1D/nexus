"""Tests for agent composition (Pipeline, Parallel, Router, Graph)."""

import pytest

import nexus
from nexus.agent import Agent
from nexus.compose import Graph, Parallel, Pipeline, Router
from nexus.context import Context
from nexus.llm.mock import MockLLM
from nexus.testing import AgentTest


@nexus.agent(name="upper")
async def upper_agent(pipeline_input: str = "", ctx: Context = None, **kw) -> str:
    text = pipeline_input or kw.get("text", "")
    return str(text).upper()


@nexus.agent(name="exclaim")
async def exclaim_agent(pipeline_input: str = "", ctx: Context = None, **kw) -> str:
    text = pipeline_input or kw.get("text", "")
    return str(text) + "!!!"


@nexus.agent(name="reverse")
async def reverse_agent(pipeline_input: str = "", ctx: Context = None, **kw) -> str:
    text = pipeline_input or kw.get("text", "")
    return str(text)[::-1]


@nexus.agent(name="counter")
async def counter_agent(pipeline_input: str = "", ctx: Context = None, **kw) -> str:
    text = pipeline_input or kw.get("text", "")
    return str(len(str(text)))


class TestPipeline:
    @pytest.mark.asyncio
    async def test_sequential(self):
        pipe = Pipeline(steps=[upper_agent, exclaim_agent], name="shout")
        ctx = Context(llm=MockLLM())
        result = await pipe(ctx=ctx, pipeline_input="hello")
        assert result == "HELLO!!!"

    @pytest.mark.asyncio
    async def test_pipeline_trace(self):
        pipe = Pipeline(steps=[upper_agent, exclaim_agent])
        ctx = Context(llm=MockLLM())
        await pipe(ctx=ctx, pipeline_input="test")
        tree = ctx.trace.print_tree()
        assert "upper" in tree
        assert "exclaim" in tree


class TestParallel:
    @pytest.mark.asyncio
    async def test_parallel_execution(self):
        par = Parallel(agents=[upper_agent, reverse_agent, counter_agent])
        ctx = Context(llm=MockLLM())
        results = await par(ctx=ctx, text="hello")
        assert results["upper"] == "HELLO"
        assert results["reverse"] == "olleh"
        assert results["counter"] == "5"


class TestRouter:
    @pytest.mark.asyncio
    async def test_function_routing(self):
        router = Router(
            routes={"upper": upper_agent, "reverse": reverse_agent},
            route_fn=lambda input, ctx: "upper" if "upper" in str(input) else "reverse",
        )
        ctx = Context(llm=MockLLM())
        result = await router(ctx=ctx, text="make upper")
        assert result == "MAKE UPPER"


class TestGraph:
    @pytest.mark.asyncio
    async def test_dag_execution(self):
        graph = Graph(name="text_pipeline")
        graph.add_node("step1", upper_agent)
        graph.add_node("step2", exclaim_agent, depends_on=["step1"])

        ctx = Context(llm=MockLLM())
        results = await graph(ctx=ctx, pipeline_input="hello")
        assert "step1" in results
        assert "step2" in results

    def test_graph_validation(self):
        graph = Graph()
        graph.add_node("a", upper_agent, depends_on=["missing"])
        errors = graph.validate()
        assert len(errors) > 0
        assert "missing" in errors[0]
