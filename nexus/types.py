"""Core type system for Nexus agents."""

from __future__ import annotations

import time
import uuid
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class Severity(str, Enum):
    """Severity levels for guardrails and quality signals."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class SpanStatus(str, Enum):
    """Status of an execution span."""
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"
    GUARDRAIL_BLOCKED = "guardrail_blocked"


class AgentSpec(BaseModel):
    """Typed specification for what an agent can do — its contract with the world."""

    name: str = Field(description="Unique agent identifier")
    description: str = Field(default="", description="What this agent does")
    version: str = Field(default="0.1.0")
    tags: list[str] = Field(default_factory=list)
    input_schema: dict[str, Any] = Field(
        default_factory=dict, description="JSON Schema for expected inputs"
    )
    output_schema: dict[str, Any] = Field(
        default_factory=dict, description="JSON Schema for outputs"
    )
    guardrails: list[str] = Field(
        default_factory=list, description="Names of guardrails to apply"
    )
    retry_policy: RetryPolicy | None = None
    timeout: float = Field(default=120.0, description="Max execution time in seconds")
    memory_enabled: bool = Field(default=True, description="Whether this agent uses memory")


class RetryPolicy(BaseModel):
    """How to retry failed agent executions."""

    max_retries: int = Field(default=3, ge=0)
    backoff_factor: float = Field(default=1.0, ge=0)
    retry_on: list[str] = Field(
        default_factory=lambda: ["AgentError", "TimeoutError"],
        description="Exception types to retry on",
    )


class AgentResult(BaseModel):
    """Result of an agent execution — wraps the output with metadata."""

    agent_name: str
    run_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    status: SpanStatus = SpanStatus.SUCCESS
    output: Any = None
    error: str | None = None
    duration_ms: float = 0.0
    tokens_used: int = 0
    memory_reads: int = 0
    memory_writes: int = 0
    guardrails_checked: int = 0
    guardrails_passed: int = 0
    trace_id: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def succeeded(self) -> bool:
        return self.status == SpanStatus.SUCCESS

    @property
    def failed(self) -> bool:
        return self.status == SpanStatus.FAILED


class MemoryEntry(BaseModel):
    """A single entry in agent memory."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    agent_name: str
    key: str
    value: Any
    entry_type: str = "semantic"  # semantic, episodic, procedural
    created_at: float = Field(default_factory=time.time)
    access_count: int = 0
    relevance_score: float = 1.0
    metadata: dict[str, Any] = Field(default_factory=dict)


class TraceEvent(BaseModel):
    """A single event in an execution trace."""

    timestamp: float = Field(default_factory=time.time)
    event_type: str  # agent_start, agent_end, llm_call, memory_read, guardrail_check, etc.
    agent_name: str = ""
    span_id: str = ""
    data: dict[str, Any] = Field(default_factory=dict)
    duration_ms: float = 0.0
