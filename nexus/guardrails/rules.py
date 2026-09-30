"""Built-in guardrail rules and the @guardrail decorator."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from nexus.errors import GuardrailViolation
from nexus.types import Severity


@dataclass
class Guardrail:
    """A named constraint that validates agent inputs or outputs."""

    name: str
    check: Callable[[Any], bool]
    message: str = ""
    severity: Severity = Severity.HIGH
    applies_to: str = "output"  # "input", "output", or "both"

    def validate(self, value: Any) -> None:
        if not self.check(value):
            raise GuardrailViolation(
                rule_name=self.name,
                message=self.message or f"Guardrail '{self.name}' check failed",
                severity=self.severity.value,
            )


def guardrail(
    name: str = "",
    severity: Severity = Severity.HIGH,
    applies_to: str = "output",
    message: str = "",
) -> Callable:
    """Decorator to create a guardrail from a function.

    @guardrail(name="no_swearing", severity=Severity.MEDIUM)
    def check_clean(output: str) -> bool:
        return not any(word in output.lower() for word in BAD_WORDS)
    """
    def decorator(fn: Callable) -> Guardrail:
        return Guardrail(
            name=name or fn.__name__,
            check=fn,
            message=message or fn.__doc__ or "",
            severity=severity,
            applies_to=applies_to,
        )
    return decorator


# ─── Built-in guardrails ──────────────────────────────────────────────


def no_pii() -> Guardrail:
    """Block outputs containing common PII patterns."""
    PII_PATTERNS = [
        r"\b\d{3}-\d{2}-\d{4}\b",          # SSN
        r"\b\d{16}\b",                       # Credit card
        r"\b[A-Z]{2}\d{6,9}\b",             # Passport
        r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b",   # Phone (US)
    ]

    def check(output: Any) -> bool:
        text = str(output)
        return not any(re.search(p, text) for p in PII_PATTERNS)

    return Guardrail(
        name="no_pii",
        check=check,
        message="Output contains potential PII (SSN, credit card, phone number)",
        severity=Severity.CRITICAL,
    )


def content_length(min_len: int = 0, max_len: int = 50000) -> Guardrail:
    """Enforce output length constraints."""
    def check(output: Any) -> bool:
        length = len(str(output))
        return min_len <= length <= max_len

    return Guardrail(
        name="content_length",
        check=check,
        message=f"Output length must be between {min_len} and {max_len} characters",
        severity=Severity.MEDIUM,
    )


def no_code_execution() -> Guardrail:
    """Block outputs that contain executable code patterns."""
    DANGEROUS = [
        r"\bexec\s*\(",
        r"\beval\s*\(",
        r"\bos\.system\s*\(",
        r"\bsubprocess\.",
        r"\b__import__\s*\(",
        r"\brm\s+-rf\b",
        r"\bdrop\s+table\b",
    ]

    def check(output: Any) -> bool:
        text = str(output).lower()
        return not any(re.search(p, text, re.IGNORECASE) for p in DANGEROUS)

    return Guardrail(
        name="no_code_execution",
        check=check,
        message="Output contains potentially dangerous code execution patterns",
        severity=Severity.CRITICAL,
    )


def output_format(expected: str) -> Guardrail:
    """Validate output format (json, markdown, python, etc.)."""
    def check(output: Any) -> bool:
        text = str(output).strip()
        if expected == "json":
            import json as json_mod
            try:
                json_mod.loads(text)
                return True
            except (json_mod.JSONDecodeError, ValueError):
                return False
        elif expected == "markdown":
            return "#" in text or "-" in text or "*" in text
        return True

    return Guardrail(
        name=f"output_format_{expected}",
        check=check,
        message=f"Output must be valid {expected}",
        severity=Severity.MEDIUM,
    )


def require_fields(*fields: str) -> Guardrail:
    """Require specific fields in dict/model output."""
    def check(output: Any) -> bool:
        if isinstance(output, dict):
            return all(f in output for f in fields)
        return all(hasattr(output, f) for f in fields)

    return Guardrail(
        name="require_fields",
        check=check,
        message=f"Output must contain fields: {', '.join(fields)}",
        severity=Severity.HIGH,
    )


def rate_limit(max_calls: int, window_seconds: float = 60.0) -> Guardrail:
    """Rate limit agent executions."""
    call_times: list[float] = []

    def check(_: Any) -> bool:
        now = time.time()
        call_times[:] = [t for t in call_times if now - t < window_seconds]
        if len(call_times) >= max_calls:
            return False
        call_times.append(now)
        return True

    return Guardrail(
        name="rate_limit",
        check=check,
        message=f"Rate limit exceeded: max {max_calls} calls per {window_seconds}s",
        severity=Severity.HIGH,
        applies_to="input",
    )
