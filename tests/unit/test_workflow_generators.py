from types import SimpleNamespace

import pytest

from fingraph_rag.domain.contracts import QuerySpec
from fingraph_rag.generation.contracts import GenerationContext
from fingraph_rag.retrieval.contracts import FusedContext
from fingraph_rag.workflow.generators import PydanticGenerationAdapter

pytestmark = pytest.mark.unit


def test_pydantic_generation_is_bounded_to_one_request_and_output_tokens():
    async def run():
        adapter = PydanticGenerationAdapter("test")
        calls = {}

        async def capture(*args, **kwargs):
            calls.update(kwargs)
            return SimpleNamespace(output=[])

        adapter.agent = SimpleNamespace(run=capture)
        context = GenerationContext(
            context=FusedContext(query=QuerySpec(question="What is revenue?"))
        )
        assert await adapter.generate(context, []) == []
        assert calls["usage_limits"].request_limit == 1
        assert calls["usage_limits"].output_tokens_limit == 2048
        assert calls["model_settings"]["max_tokens"] == 2048

    import asyncio

    asyncio.run(run())
