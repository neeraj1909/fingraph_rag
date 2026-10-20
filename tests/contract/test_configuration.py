import pytest
from pydantic import ValidationError

from fingraph_rag.settings import Settings


@pytest.mark.contract
def test_production_rejects_default_secret() -> None:
    with pytest.raises(ValidationError, match="must be changed"):
        Settings(app_env="prod")


@pytest.mark.contract
def test_test_environment_accepts_local_defaults() -> None:
    settings = Settings(app_env="test")

    assert settings.max_hops == 4
    assert settings.cypher_repair_retries == 2
