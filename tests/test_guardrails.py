"""Tests for guardrails."""

import pytest

from nexus.errors import GuardrailViolation
from nexus.guardrails import GuardrailEngine, content_length, no_code_execution, no_pii, require_fields
from nexus.guardrails.rules import Guardrail, guardrail
from nexus.types import Severity


class TestBuiltinGuardrails:
    def test_no_pii_clean(self):
        g = no_pii()
        g.validate("This is a normal text with no PII")

    def test_no_pii_ssn(self):
        g = no_pii()
        with pytest.raises(GuardrailViolation, match="PII"):
            g.validate("My SSN is 123-45-6789")

    def test_no_pii_credit_card(self):
        g = no_pii()
        with pytest.raises(GuardrailViolation):
            g.validate("Card: 1234567890123456")

    def test_content_length_valid(self):
        g = content_length(min_len=5, max_len=100)
        g.validate("Hello, World!")

    def test_content_length_too_short(self):
        g = content_length(min_len=10, max_len=100)
        with pytest.raises(GuardrailViolation):
            g.validate("Hi")

    def test_no_code_execution_clean(self):
        g = no_code_execution()
        g.validate("Calculate the sum of 1 + 2")

    def test_no_code_execution_dangerous(self):
        g = no_code_execution()
        with pytest.raises(GuardrailViolation):
            g.validate("Run os.system('rm -rf /')")

    def test_require_fields_dict(self):
        g = require_fields("name", "age")
        g.validate({"name": "Alice", "age": 30})

    def test_require_fields_missing(self):
        g = require_fields("name", "age")
        with pytest.raises(GuardrailViolation):
            g.validate({"name": "Alice"})


class TestGuardrailDecorator:
    def test_custom_guardrail(self):
        @guardrail(name="no_swearing", severity=Severity.MEDIUM)
        def check_clean(output):
            """No bad words allowed."""
            return "badword" not in output.lower()

        check_clean.validate("This is fine")
        with pytest.raises(GuardrailViolation):
            check_clean.validate("This has a badword in it")


class TestGuardrailEngine:
    def test_engine_checks_output(self):
        engine = GuardrailEngine([
            content_length(min_len=1, max_len=1000),
            no_pii(),
        ])
        engine.check_output("Normal text")
        assert engine.checks_run == 2
        assert engine.checks_passed == 2

    def test_engine_collects_violations(self):
        engine = GuardrailEngine([no_pii()])
        violations = engine.check_all("SSN: 123-45-6789")
        assert len(violations) == 1
        assert "PII" in str(violations[0])
