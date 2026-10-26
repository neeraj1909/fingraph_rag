import pytest
from tests.unit.test_calculator import metric

from fingraph_rag.domain.contracts import QuerySpec, RelationshipAssertion
from fingraph_rag.generation.citation_renderer import (
    CitationError,
    CitationResolver,
    render_citation,
)
from fingraph_rag.generation.generator import ExtractiveGenerator, prepare_generation_context
from fingraph_rag.guardrails.pipeline import GuardrailPipeline
from fingraph_rag.retrieval.contracts import FusedContext, GraphPath, TextEvidence

pytestmark = pytest.mark.unit


def generation_fixture(calculation=False):
    paths, evidence = [], []
    years = [2024, 2025] if calculation else [2025]
    for year in years:
        observation = metric("120" if year == 2025 else "100", year)
        item = TextEvidence(
            evidence_id=f"e-{year}",
            text=f"FY{year} revenue: {observation.value_decimal} million USD.",
            locator=observation.locator,
            temporal=observation.temporal,
            entity_ids=["acme"],
            metric_key="revenue",
        )
        evidence.append(item)
        rel = RelationshipAssertion(
            assertion_id=f"rel-{year}",
            subject_entity_id="acme",
            predicate="REPORTED_METRIC",
            object_entity_id=observation.metric_id,
            properties={"metric_observation": observation.model_dump(mode="json")},
            temporal=observation.temporal,
            source_doc_id=item.locator.doc_id,
            extraction_run_id="extract-1",
            locators=[item.locator],
            confidence=1,
        )
        paths.append(
            GraphPath(
                path_id=f"path-{year}",
                entity_ids=["acme", observation.metric_id],
                relationships=[rel],
            )
        )
    context = FusedContext(
        query=QuerySpec(
            question="Calculate revenue growth" if calculation else "What is revenue?",
            needs_calculation=calculation,
        ),
        graph_paths=paths,
        text_evidence=evidence,
    )
    generation = prepare_generation_context(context)
    return generation


def candidate(generation):
    return ExtractiveGenerator().generate(generation)[0]


def test_resolved_citation_uses_stored_locator_and_directed_graph_path():
    generation = generation_fixture()
    claim = candidate(generation)
    citation = CitationResolver().resolve(claim, generation)[0]
    assert citation.document == generation.context.text_evidence[0].locator
    assert citation.relation_path == "(acme)-[REPORTED_METRIC:rel-2025]->(revenue-2025)"
    assert "[Doc: doc-2025" in render_citation(citation)
    result = GuardrailPipeline().verify_claims([claim], generation)[0]
    assert result.verification == "verified"


def test_wrong_document_and_missing_path_fail_closed():
    generation = generation_fixture()
    claim = candidate(generation)
    with pytest.raises(CitationError, match="unresolved_citation"):
        CitationResolver().resolve(
            claim.model_copy(update={"graph_path_ids": ["made-up"]}), generation
        )
    generation.context.graph_paths[0].relationships[0].locators[0] = (
        generation.context.graph_paths[0]
        .relationships[0]
        .locators[0]
        .model_copy(update={"page": 9})
    )
    with pytest.raises(CitationError, match="document_graph_source_mismatch"):
        CitationResolver().resolve(claim, generation)


def test_graph_source_document_mismatch_fails_closed():
    generation = generation_fixture()
    claim = candidate(generation)
    generation.context.graph_paths[0].relationships[0].source_doc_id = "forged-doc"
    with pytest.raises(CitationError, match="document_graph_source_mismatch"):
        CitationResolver().resolve(claim, generation)


def test_calculation_derives_real_observations_and_cites_every_input():
    generation = generation_fixture(calculation=True)
    assert set(generation.metrics) == {"e-2024", "e-2025"}
    assert generation.calculation_error is None
    claim = candidate(generation)
    assert claim.text == "yoy_growth: 20.00 %."
    result = GuardrailPipeline().verify_claims([claim], generation)[0]
    assert result.verification == "verified"
    assert {c.document.doc_id for c in result.citations} == {"doc-2024", "doc-2025"}
    assert all(c.calculation_id for c in result.citations)
    broken = claim.model_copy(update={"evidence_ids": ["e-2025"], "graph_path_ids": ["path-2025"]})
    assert GuardrailPipeline().verify_claims([broken], generation)[0].verification == "abstained"


def test_calculation_citation_requires_reproducibility_request():
    generation = generation_fixture(calculation=True)
    claim = candidate(generation)
    generation.calculation_requests.clear()
    with pytest.raises(CitationError, match="unresolved_calculation_request"):
        CitationResolver().resolve(claim, generation)


def test_ambiguous_calculation_abstains_without_substituting_raw_extracts():
    generation = generation_fixture()
    generation.context.query.needs_calculation = True
    generation = prepare_generation_context(generation.context)
    assert generation.calculation_error == "ambiguous_calculation_formula"
    assert ExtractiveGenerator().generate(generation) == []


def test_no_graph_evidence_never_produces_factual_claim():
    generation = generation_fixture()
    generation.context.graph_paths.clear()
    assert ExtractiveGenerator().generate(generation) == []
