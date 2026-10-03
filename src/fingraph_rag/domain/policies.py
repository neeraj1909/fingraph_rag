"""Small, framework-neutral policy value objects."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ExecutionPolicy:
    """Hard limits shared by query planning and later guardrails."""

    max_hops: int = 4
    max_context_tokens: int = 12_000
    max_query_seconds: int = 30
    max_cypher_repair_retries: int = 2

    def __post_init__(self) -> None:
        if not 0 <= self.max_hops <= 4:
            raise ValueError("max_hops must be between 0 and 4")
        if self.max_context_tokens < 256:
            raise ValueError("max_context_tokens must be at least 256")
        if self.max_query_seconds < 1:
            raise ValueError("max_query_seconds must be positive")
        if not 0 <= self.max_cypher_repair_retries <= 2:
            raise ValueError("max_cypher_repair_retries must be between 0 and 2")


DEFAULT_EXECUTION_POLICY = ExecutionPolicy()
