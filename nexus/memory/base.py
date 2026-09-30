"""Abstract memory interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from nexus.types import MemoryEntry


class MemoryStore(ABC):
    """Interface for agent memory backends."""

    @abstractmethod
    async def remember(self, agent_name: str, key: str, value: Any, entry_type: str = "semantic") -> str:
        """Store a memory entry. Returns the entry ID."""

    @abstractmethod
    async def recall(self, agent_name: str, key: str, limit: int = 5) -> list[MemoryEntry]:
        """Retrieve memory entries matching a key (fuzzy search)."""

    @abstractmethod
    async def recall_recent(self, agent_name: str, limit: int = 10) -> list[MemoryEntry]:
        """Retrieve the most recent memory entries."""

    @abstractmethod
    async def forget(self, agent_name: str, key: str) -> int:
        """Delete memory entries matching a key. Returns count deleted."""

    @abstractmethod
    async def count(self, agent_name: str) -> int:
        """Count total memory entries for an agent."""

    @abstractmethod
    async def clear(self, agent_name: str) -> None:
        """Clear all memory for an agent."""
