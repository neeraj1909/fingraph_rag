"""Resolve citations from retained evidence, never from model-authored locators."""

import hashlib

from fingraph_rag.domain.contracts import Citation
from fingraph_rag.generation.contracts import CandidateClaim, GenerationContext
from fingraph_rag.retrieval.contracts import GraphPath


class CitationError(ValueError):
    pass


def relation_path(path: GraphPath) -> str:
    if len(path.entity_ids) != len(path.relationships) + 1:
        raise CitationError("invalid_graph_path")
    pieces = [f"({path.entity_ids[0]})"]
    for index, rel in enumerate(path.relationships):
        direction = "->" if rel.subject_entity_id == path.entity_ids[index] else "-"
        prefix = "-" if direction == "->" else "<-"
        pieces.append(
            f"{prefix}[{rel.predicate}:{rel.assertion_id}]{direction}({path.entity_ids[index + 1]})"
        )
    return "".join(pieces)


class CitationResolver:
    def resolve(self, claim: CandidateClaim, generation: GenerationContext) -> list[Citation]:
        evidence = generation.evidence()
        paths = {p.path_id: p for p in generation.context.graph_paths}
        if not claim.evidence_ids or not claim.graph_path_ids:
            raise CitationError("missing_dual_citations")
        if len(set(claim.evidence_ids)) != len(claim.evidence_ids) or len(
            set(claim.graph_path_ids)
        ) != len(claim.graph_path_ids):
            raise CitationError("duplicate_citations")
        if len(set(claim.calculation_ids)) != len(claim.calculation_ids):
            raise CitationError("duplicate_calculations")
        if any(key not in evidence for key in claim.evidence_ids) or any(
            key not in paths for key in claim.graph_path_ids
        ):
            raise CitationError("unresolved_citation")
        selected_paths = [paths[key] for key in claim.graph_path_ids]
        citations = []
        used_paths = set()
        for evidence_id in claim.evidence_ids:
            item = evidence[evidence_id]
            matching = [
                p
                for p in selected_paths
                if any(
                    locator == item.locator
                    and rel.source_doc_id == item.locator.doc_id
                    and rel.temporal == item.temporal
                    and rel.status == "asserted"
                    for rel in p.relationships
                    for locator in rel.locators
                )
            ]
            if not matching:
                raise CitationError("document_graph_source_mismatch")
            path = matching[0]
            used_paths.add(path.path_id)
            digest = hashlib.sha256(
                f"{claim.claim_id}:{evidence_id}:{path.path_id}".encode()
            ).hexdigest()
            citations.append(
                Citation(
                    citation_id=digest,
                    document=item.locator,
                    graph_path_id=path.path_id,
                    relation_path=relation_path(path),
                )
            )
        if used_paths != set(claim.graph_path_ids):
            raise CitationError("unrelated_graph_citation")
        if claim.calculation_ids:
            input_ids = set()
            for key in claim.calculation_ids:
                result = generation.calculations.get(key)
                if result is None:
                    raise CitationError("unresolved_calculation")
                request = generation.calculation_requests.get(key)
                if request is None:
                    raise CitationError("unresolved_calculation_request")
                if list(result.input_evidence_ids) != list(request.inputs):
                    raise CitationError("calculation_input_mismatch")
                input_ids.update(result.input_evidence_ids)
            if not input_ids.issubset(set(claim.evidence_ids)):
                raise CitationError("missing_calculation_input_citations")
            # Each calculation has a source citation for every input, not just its output.
            citations = [
                citation.model_copy(update={"calculation_id": key})
                for key in claim.calculation_ids
                for citation in citations
            ]
        return citations


def render_citation(citation: Citation) -> str:
    locator = citation.document
    location = " / ".join(
        str(part)
        for part in (
            locator.section,
            f"page {locator.page}" if locator.page else None,
            locator.table_id,
            locator.cell_range,
        )
        if part
    )
    calculation = f" [Calc: {citation.calculation_id}]" if citation.calculation_id else ""
    return f"[Doc: {locator.doc_id}, {locator.filing_date}, {location}] [Graph: {citation.relation_path}]{calculation}"
