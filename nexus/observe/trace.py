"""Execution tracing — full visibility into every agent decision."""

from __future__ import annotations

import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Generator

from nexus.types import SpanStatus, TraceEvent


@dataclass
class Span:
    """A single operation within a trace — an agent call, LLM request, memory access, etc."""

    span_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str = ""
    agent_name: str = ""
    status: SpanStatus = SpanStatus.RUNNING
    start_time: float = field(default_factory=time.time)
    end_time: float = 0.0
    input_data: Any = None
    output_data: Any = None
    error: str | None = None
    children: list[Span] = field(default_factory=list)
    events: list[TraceEvent] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def duration_ms(self) -> float:
        if self.end_time:
            return (self.end_time - self.start_time) * 1000
        return (time.time() - self.start_time) * 1000

    def add_event(self, event_type: str, data: dict[str, Any] | None = None) -> None:
        self.events.append(
            TraceEvent(
                event_type=event_type,
                agent_name=self.agent_name,
                span_id=self.span_id,
                data=data or {},
            )
        )

    def succeed(self, output: Any = None) -> None:
        self.status = SpanStatus.SUCCESS
        self.output_data = output
        self.end_time = time.time()

    def fail(self, error: str) -> None:
        self.status = SpanStatus.FAILED
        self.error = error
        self.end_time = time.time()

    def to_dict(self) -> dict[str, Any]:
        return {
            "span_id": self.span_id,
            "name": self.name,
            "agent_name": self.agent_name,
            "status": self.status.value,
            "duration_ms": round(self.duration_ms, 2),
            "error": self.error,
            "children": [c.to_dict() for c in self.children],
            "events": [
                {"type": e.event_type, "data": e.data, "time": e.timestamp}
                for e in self.events
            ],
            "metadata": self.metadata,
        }


@dataclass
class Trace:
    """A complete execution trace — a tree of spans representing an agent workflow."""

    trace_id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    root_span: Span | None = None
    spans: list[Span] = field(default_factory=list)
    _span_stack: list[Span] = field(default_factory=list, repr=False)

    @contextmanager
    def span(self, name: str, agent_name: str = "", **metadata: Any) -> Generator[Span, None, None]:
        """Context manager to create and track a span."""
        s = Span(name=name, agent_name=agent_name, metadata=metadata)

        if self._span_stack:
            self._span_stack[-1].children.append(s)
        elif self.root_span is None:
            self.root_span = s

        self._span_stack.append(s)
        self.spans.append(s)

        try:
            yield s
            if s.status == SpanStatus.RUNNING:
                s.succeed()
        except Exception as e:
            s.fail(str(e))
            raise
        finally:
            self._span_stack.pop()

    @property
    def duration_ms(self) -> float:
        return self.root_span.duration_ms if self.root_span else 0.0

    @property
    def status(self) -> SpanStatus:
        if self.root_span:
            return self.root_span.status
        return SpanStatus.SUCCESS if not any(s.status == SpanStatus.FAILED for s in self.spans) else SpanStatus.FAILED

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "status": self.status.value,
            "duration_ms": round(self.duration_ms, 2),
            "span_count": len(self.spans),
            "root": self.root_span.to_dict() if self.root_span else None,
        }

    def print_tree(self, indent: int = 0) -> str:
        """Render the trace as a human-readable tree."""
        lines: list[str] = []
        if self.root_span:
            self._print_span(self.root_span, lines, indent)
        return "\n".join(lines)

    def _print_span(self, span: Span, lines: list[str], indent: int) -> None:
        icon = {"success": "✅", "failed": "❌", "running": "⏳", "skipped": "⏭️", "guardrail_blocked": "🛑"}.get(
            span.status.value, "❓"
        )
        prefix = "  " * indent
        lines.append(f"{prefix}{icon} {span.name} [{span.duration_ms:.0f}ms]")
        if span.error:
            lines.append(f"{prefix}  └─ ERROR: {span.error}")
        for child in span.children:
            self._print_span(child, lines, indent + 1)
