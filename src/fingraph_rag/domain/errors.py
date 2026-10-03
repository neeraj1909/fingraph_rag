"""Stable domain error taxonomy used across adapters and workflows."""


class FinGraphError(Exception):
    """Base class for expected application errors."""


class ValidationError(FinGraphError):
    """An input or contract value is invalid."""


class PolicyViolation(FinGraphError):
    """A safety, compliance, or execution policy rejected an operation."""


class EvidenceMissing(FinGraphError):
    """A claim or operation lacks the evidence required to proceed."""


class TemporalMismatch(FinGraphError):
    """Evidence and query periods cannot be reconciled safely."""


class BudgetExceeded(FinGraphError):
    """A bounded execution budget was exceeded."""


class DependencyUnavailable(FinGraphError):
    """A required external dependency is unavailable."""


class AbstentionRequired(FinGraphError):
    """The system must abstain instead of producing an unsupported answer."""
