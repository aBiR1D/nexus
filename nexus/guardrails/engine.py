"""Guardrail evaluation engine."""

from __future__ import annotations

from typing import Any

from nexus.errors import GuardrailViolation
from nexus.guardrails.rules import Guardrail


class GuardrailEngine:
    """Evaluates guardrails against agent inputs/outputs."""

    def __init__(self, guardrails: list[Guardrail] | None = None):
        self.guardrails = guardrails or []
        self.checks_run = 0
        self.checks_passed = 0
        self.violations: list[GuardrailViolation] = []

    def add(self, guardrail: Guardrail) -> None:
        self.guardrails.append(guardrail)

    def check_input(self, value: Any) -> None:
        """Validate input against input guardrails. Raises on violation."""
        for g in self.guardrails:
            if g.applies_to in ("input", "both"):
                self.checks_run += 1
                try:
                    g.validate(value)
                    self.checks_passed += 1
                except GuardrailViolation as e:
                    self.violations.append(e)
                    raise

    def check_output(self, value: Any) -> None:
        """Validate output against output guardrails. Raises on violation."""
        for g in self.guardrails:
            if g.applies_to in ("output", "both"):
                self.checks_run += 1
                try:
                    g.validate(value)
                    self.checks_passed += 1
                except GuardrailViolation as e:
                    self.violations.append(e)
                    raise

    def check_all(self, value: Any, phase: str = "output") -> list[GuardrailViolation]:
        """Run all guardrails and collect violations (non-raising)."""
        violations = []
        for g in self.guardrails:
            if g.applies_to in (phase, "both"):
                self.checks_run += 1
                try:
                    g.validate(value)
                    self.checks_passed += 1
                except GuardrailViolation as e:
                    violations.append(e)
        return violations

    @property
    def stats(self) -> dict:
        return {
            "total_guardrails": len(self.guardrails),
            "checks_run": self.checks_run,
            "checks_passed": self.checks_passed,
            "violations": len(self.violations),
        }
