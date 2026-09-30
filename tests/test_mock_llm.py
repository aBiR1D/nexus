"""Tests for MockLLM."""

import pytest

from pydantic import BaseModel

from nexus.llm.mock import MockLLM


class SampleOutput(BaseModel):
    answer: str
    score: int


class TestMockLLM:
    @pytest.mark.asyncio
    async def test_sequential_responses(self):
        mock = MockLLM(responses=["First", "Second", "Third"])
        assert await mock.generate("q1") == "First"
        assert await mock.generate("q2") == "Second"
        assert await mock.generate("q3") == "Third"

    @pytest.mark.asyncio
    async def test_default_response(self):
        mock = MockLLM(default_response="Default!")
        assert await mock.generate("anything") == "Default!"

    @pytest.mark.asyncio
    async def test_response_map(self):
        mock = MockLLM(response_map={
            "hello": "Hi there!",
            "bye": "Goodbye!",
        })
        assert await mock.generate("Say hello") == "Hi there!"
        assert await mock.generate("Say bye") == "Goodbye!"

    @pytest.mark.asyncio
    async def test_structured_output(self):
        mock = MockLLM(responses=[SampleOutput(answer="42", score=10)])
        result = await mock.generate_structured("What?", SampleOutput)
        assert result.answer == "42"
        assert result.score == 10

    @pytest.mark.asyncio
    async def test_classify(self):
        mock = MockLLM(responses=["positive"])
        result = await mock.classify("Great product!", ["positive", "negative"])
        assert result == "positive"

    def test_assert_called(self):
        mock = MockLLM()
        mock._record_call("test prompt")
        mock.assert_called()
        mock.assert_called(times=1)

    def test_assert_prompt_contains(self):
        mock = MockLLM()
        mock._record_call("Write a Python function")
        mock.assert_prompt_contains("Python")

    @pytest.mark.asyncio
    async def test_reset(self):
        mock = MockLLM(responses=["A", "B"])
        await mock.generate("q1")
        mock.reset()
        assert mock.call_count == 0
        assert await mock.generate("q1") == "A"
