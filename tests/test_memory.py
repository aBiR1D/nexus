"""Tests for agent memory."""

import pytest

from nexus.memory.local import LocalMemory


@pytest.fixture
def memory(tmp_path):
    db_path = tmp_path / "test_memory.db"
    mem = LocalMemory(db_path)
    yield mem
    mem.close()


class TestLocalMemory:
    @pytest.mark.asyncio
    async def test_remember_and_recall(self, memory):
        await memory.remember("agent1", "greeting", "Hello World")
        entries = await memory.recall("agent1", "greeting")
        assert len(entries) == 1
        assert entries[0].value == "Hello World"

    @pytest.mark.asyncio
    async def test_recall_fuzzy_match(self, memory):
        await memory.remember("agent1", "python patterns", "Use list comprehensions")
        await memory.remember("agent1", "java patterns", "Use streams")

        entries = await memory.recall("agent1", "python")
        assert len(entries) == 1
        assert "list comprehensions" in str(entries[0].value)

    @pytest.mark.asyncio
    async def test_recall_recent(self, memory):
        await memory.remember("agent1", "key1", "val1")
        await memory.remember("agent1", "key2", "val2")
        await memory.remember("agent1", "key3", "val3")

        entries = await memory.recall_recent("agent1", limit=2)
        assert len(entries) == 2

    @pytest.mark.asyncio
    async def test_forget(self, memory):
        await memory.remember("agent1", "temp", "temporary data")
        assert await memory.count("agent1") == 1
        deleted = await memory.forget("agent1", "temp")
        assert deleted == 1
        assert await memory.count("agent1") == 0

    @pytest.mark.asyncio
    async def test_clear(self, memory):
        await memory.remember("agent1", "k1", "v1")
        await memory.remember("agent1", "k2", "v2")
        assert await memory.count("agent1") == 2
        await memory.clear("agent1")
        assert await memory.count("agent1") == 0

    @pytest.mark.asyncio
    async def test_agent_isolation(self, memory):
        await memory.remember("agent1", "key", "val1")
        await memory.remember("agent2", "key", "val2")
        assert await memory.count("agent1") == 1
        assert await memory.count("agent2") == 1

    @pytest.mark.asyncio
    async def test_json_values(self, memory):
        await memory.remember("agent1", "config", {"model": "gpt-4", "temp": 0.7})
        entries = await memory.recall("agent1", "config")
        assert entries[0].value == {"model": "gpt-4", "temp": 0.7}
