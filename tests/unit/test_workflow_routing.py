import pytest

from fingraph_rag.workflow.routing import verification_route

pytestmark = pytest.mark.unit


def test_shared_repair_budget_and_terminal_routing():
    assert verification_route({"terminal": "answer", "repairs": 0}) == "finish"
    assert verification_route({"terminal": "abstain", "repairs": 0, "candidates": [1]}) == "repair"
    assert verification_route({"terminal": "abstain", "repairs": 2, "candidates": [1]}) == "finish"
    assert verification_route({"terminal": "abstain", "repairs": 0, "candidates": []}) == "finish"
