import pytest

from fingraph_rag.vector.fusion import RrfFusion

pytestmark = pytest.mark.unit


def test_rrf_skips_malformed_records_without_crashing():
    result = RrfFusion().fuse(
        [[{}, {"evidence_id": ""}, {"evidence_id": "valid", "score": 1.0}, None]], 5
    )
    assert [record["evidence_id"] for record in result] == ["valid"]
