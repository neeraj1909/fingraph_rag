"""Validated runtime configuration for the foundation slice."""

from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Environment-backed configuration with fail-closed production checks."""

    model_config = SettingsConfigDict(
        env_file=(".env",),
        env_prefix="FINGRAPH_",
        extra="ignore",
        case_sensitive=False,
    )

    app_env: Literal["dev", "test", "prod"] = "dev"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    postgres_dsn: str = "postgresql://fingraph:fingraph_dev@localhost:5432/fingraph"
    redis_url: str = "redis://localhost:6379/0"
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: SecretStr = SecretStr("change-me-before-use")
    neo4j_read_user: str | None = None
    neo4j_read_password: SecretStr | None = None
    qdrant_url: str = "http://localhost:6333"

    max_hops: int = Field(default=4, ge=0, le=4)
    max_context_tokens: int = Field(default=12_000, ge=256, le=100_000)
    max_query_seconds: int = Field(default=30, ge=1, le=300)
    cypher_repair_retries: int = Field(default=2, ge=0, le=2)

    @model_validator(mode="after")
    def validate_runtime_secrets(self) -> "Settings":
        if self.app_env == "prod":
            if not self.neo4j_read_user or not self.neo4j_read_password:
                raise ValueError(
                    "FINGRAPH_NEO4J_READ_USER and FINGRAPH_NEO4J_READ_PASSWORD "
                    "must be changed in prod"
                )
            if self.neo4j_read_user == self.neo4j_user:
                raise ValueError("production graph reader credentials must be separate")
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide validated settings instance."""

    return Settings()
