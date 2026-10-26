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


@pytest.mark.contract
def test_production_reader_credentials_do_not_require_admin_password() -> None:
    settings = Settings(
        app_env="prod",
        neo4j_read_user="reader",
        neo4j_read_password="read-only-secret",
    )

    assert settings.neo4j_password.get_secret_value() == "change-me-before-use"
    assert settings.neo4j_read_user == "reader"
