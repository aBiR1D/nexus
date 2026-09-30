"""Nexus exception hierarchy."""


class NexusError(Exception):
    """Base exception for all Nexus errors."""


class AgentError(NexusError):
    """Error during agent execution."""


class GuardrailViolation(NexusError):
    """A guardrail constraint was violated."""

    def __init__(self, rule_name: str, message: str, severity: str = "error"):
        self.rule_name = rule_name
        self.severity = severity
        super().__init__(f"Guardrail '{rule_name}' violated: {message}")


class RetryExhausted(AgentError):
    """All retry attempts failed."""

    def __init__(self, agent_name: str, attempts: int, last_error: Exception):
        self.agent_name = agent_name
        self.attempts = attempts
        self.last_error = last_error
        super().__init__(
            f"Agent '{agent_name}' failed after {attempts} attempts: {last_error}"
        )


class MemoryError(NexusError):
    """Error accessing agent memory."""


class CompositionError(NexusError):
    """Error in agent composition (pipeline, router, etc.)."""
