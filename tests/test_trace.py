"""Tests for execution tracing."""

import pytest

from nexus.observe.trace import Span, Trace
from nexus.types import SpanStatus


class TestTrace:
    def test_span_context_manager(self):
        trace = Trace()
        with trace.span("test_op") as s:
            s.add_event("doing_work", {"step": 1})
        assert s.status == SpanStatus.SUCCESS
        assert s.duration_ms >= 0

    def test_span_failure(self):
        trace = Trace()
        with pytest.raises(ValueError):
            with trace.span("failing_op") as s:
                raise ValueError("test error")
        assert s.status == SpanStatus.FAILED
        assert "test error" in s.error

    def test_nested_spans(self):
        trace = Trace()
        with trace.span("parent") as parent:
            with trace.span("child1") as c1:
                pass
            with trace.span("child2") as c2:
                pass
        assert len(parent.children) == 2
        assert parent.children[0].name == "child1"
        assert parent.children[1].name == "child2"

    def test_trace_to_dict(self):
        trace = Trace()
        with trace.span("root") as s:
            s.add_event("test", {"key": "value"})
        d = trace.to_dict()
        assert "trace_id" in d
        assert d["status"] == "success"
        assert d["root"]["name"] == "root"

    def test_print_tree(self):
        trace = Trace()
        with trace.span("root") as _:
            with trace.span("step1") as _:
                pass
            with trace.span("step2") as _:
                pass
        tree = trace.print_tree()
        assert "root" in tree
        assert "step1" in tree
        assert "step2" in tree
        assert "✅" in tree

    def test_print_tree_with_error(self):
        trace = Trace()
        try:
            with trace.span("root") as _:
                with trace.span("failing") as _:
                    raise RuntimeError("boom")
        except RuntimeError:
            pass
        tree = trace.print_tree()
        assert "❌" in tree
        assert "boom" in tree
