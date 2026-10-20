"""Model gateway protocol; no provider SDK is imported in the foundation."""

from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field


class ModelRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt: str = Field(min_length=1)
    output_schema: str = Field(min_length=1)
    temperature: float = Field(default=0, ge=0, le=2)


class ModelResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_name: str = Field(min_length=1)
    structured_output: dict[str, object]
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)


@runtime_checkable
class ModelGateway(Protocol):
    async def generate(self, request: ModelRequest) -> ModelResponse: ...
