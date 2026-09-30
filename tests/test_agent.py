"""Tests for the @agent decorator and Agent execution."""

import pytest

import nexus
from nexus.context import Context
from nexus.llm.mock import MockLLM
from nexus.testing import AgentTest


@nexus.agent(name="echo")
async def echo_agent(message: str, ctx: Context) -> str:
    """Simply returns the message."""
    return message


@nexus.agent(name="llm_greeter")
async def llm_greeter(name: str, ctx: Context) -> str:
    """Uses LLM to generate a greeting."""
    return await ctx.llm.generate(f"Greet {name}")


class TestAgentDecorator:
    @pytest.mark.asyncio
    async def test_basic_agent(self):
        result = await echo_agent(message="hello", ctx=Context(llm=MockLLM()))
        assert result == "hello"

    @pytest.mark.asyncio
    async def test_agent_has_spec(self):
        assert echo_agent.name == "echo"
        assert echo_agent.spec.name == "echo"

    @pytest.mark.asyncio
    async def test_agent_with_mock_llm(self):
        test = AgentTest(mock_responses=["Hello, World!"])
        result = await test.run(llm_greeter, name="World")
        assert result.succeeded
        assert result.output == "Hello, World!"
        test.mock.assert_called(times=1)
        test.mock.assert_prompt_contains("World")

    @pytest.mark.asyncio
    async def test_agent_creates_context_if_missing(self):
        mock = MockLLM(responses=["Hi!"])
        ctx = Context(llm=mock)
        result = await llm_greeter(name="Test", ctx=ctx)
        assert result == "Hi!"

    @pytest.mark.asyncio
    async def test_agent_result_metadata(self):
        test = AgentTest(mock_responses=["Greetings!"])
        result = await test.run(llm_greeter, name="Alice")
        assert result.agent_name == "llm_greeter"
        assert result.status.value == "success"
        assert result.duration_ms >= 0
