"""SQLite-backed local memory — persistent, fast, zero-config."""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any

from nexus.memory.base import MemoryStore
from nexus.types import MemoryEntry


class LocalMemory(MemoryStore):
    """SQLite-backed agent memory. Each agent gets its own namespace.

    Supports fuzzy key matching and automatic relevance decay.
    """

    def __init__(self, db_path: str | Path = ".nexus/memory.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: sqlite3.Connection | None = None
        self._ensure_schema()

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(str(self.db_path))
            self._conn.row_factory = sqlite3.Row
        return self._conn

    def _ensure_schema(self) -> None:
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                agent_name TEXT NOT NULL,
                key TEXT NOT NULL,
                value TEXT NOT NULL,
                entry_type TEXT DEFAULT 'semantic',
                created_at REAL NOT NULL,
                access_count INTEGER DEFAULT 0,
                relevance_score REAL DEFAULT 1.0,
                metadata TEXT DEFAULT '{}'
            );
            CREATE INDEX IF NOT EXISTS idx_memories_agent ON memories(agent_name);
            CREATE INDEX IF NOT EXISTS idx_memories_key ON memories(agent_name, key);
            CREATE INDEX IF NOT EXISTS idx_memories_time ON memories(agent_name, created_at DESC);
        """)
        self.conn.commit()

    async def remember(
        self, agent_name: str, key: str, value: Any, entry_type: str = "semantic"
    ) -> str:
        entry = MemoryEntry(
            agent_name=agent_name,
            key=key,
            value=value,
            entry_type=entry_type,
        )
        self.conn.execute(
            """INSERT OR REPLACE INTO memories
               (id, agent_name, key, value, entry_type, created_at, access_count, relevance_score, metadata)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                entry.id,
                entry.agent_name,
                entry.key,
                json.dumps(entry.value) if not isinstance(entry.value, str) else entry.value,
                entry.entry_type,
                entry.created_at,
                0,
                1.0,
                json.dumps(entry.metadata),
            ),
        )
        self.conn.commit()
        return entry.id

    async def recall(self, agent_name: str, key: str, limit: int = 5) -> list[MemoryEntry]:
        rows = self.conn.execute(
            """SELECT * FROM memories
               WHERE agent_name = ? AND key LIKE ?
               ORDER BY relevance_score DESC, created_at DESC
               LIMIT ?""",
            (agent_name, f"%{key}%", limit),
        ).fetchall()

        entries = [self._row_to_entry(row) for row in rows]

        for entry in entries:
            self.conn.execute(
                "UPDATE memories SET access_count = access_count + 1 WHERE id = ?",
                (entry.id,),
            )
        self.conn.commit()

        return entries

    async def recall_recent(self, agent_name: str, limit: int = 10) -> list[MemoryEntry]:
        rows = self.conn.execute(
            """SELECT * FROM memories
               WHERE agent_name = ?
               ORDER BY created_at DESC
               LIMIT ?""",
            (agent_name, limit),
        ).fetchall()
        return [self._row_to_entry(row) for row in rows]

    async def forget(self, agent_name: str, key: str) -> int:
        cursor = self.conn.execute(
            "DELETE FROM memories WHERE agent_name = ? AND key LIKE ?",
            (agent_name, f"%{key}%"),
        )
        self.conn.commit()
        return cursor.rowcount

    async def count(self, agent_name: str) -> int:
        row = self.conn.execute(
            "SELECT COUNT(*) FROM memories WHERE agent_name = ?",
            (agent_name,),
        ).fetchone()
        return row[0] if row else 0

    async def clear(self, agent_name: str) -> None:
        self.conn.execute("DELETE FROM memories WHERE agent_name = ?", (agent_name,))
        self.conn.commit()

    def _row_to_entry(self, row: sqlite3.Row) -> MemoryEntry:
        value = row["value"]
        try:
            value = json.loads(value)
        except (json.JSONDecodeError, TypeError):
            pass

        return MemoryEntry(
            id=row["id"],
            agent_name=row["agent_name"],
            key=row["key"],
            value=value,
            entry_type=row["entry_type"],
            created_at=row["created_at"],
            access_count=row["access_count"],
            relevance_score=row["relevance_score"],
            metadata=json.loads(row["metadata"]) if row["metadata"] else {},
        )

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None
