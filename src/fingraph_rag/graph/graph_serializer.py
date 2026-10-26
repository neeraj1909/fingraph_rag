"""Lossless JSON payloads alongside the scalar fields needed for graph search."""

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass

from fingraph_rag.domain.contracts import CanonicalEntity, ExtractionEnvelope, RelationshipAssertion
from fingraph_rag.domain.errors import ValidationError
from fingraph_rag.graph.ports import GraphStore


def entity_record(entity: CanonicalEntity) -> dict[str, object]:
    return {
        "entity_id": entity.entity_id,
        "canonical_name": entity.canonical_name,
        "entity_type": entity.entity_type,
        "payload": entity.model_dump_json(),
    }


def envelope_records(
    envelope: ExtractionEnvelope,
    entities: Sequence[CanonicalEntity],
    issuer_entity_id: str | None = None,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    known = {entity.entity_id for entity in entities}
    if len(known) != len(entities):
        raise ValidationError("canonical entity IDs must be unique")
    relationships = []
    for assertion in envelope.relationships:
        if assertion.subject_entity_id not in known or assertion.object_entity_id not in known:
            raise ValidationError("relationship endpoints must have canonical entity records")
        if assertion.source_doc_id != envelope.document_id:
            raise ValidationError("relationship document must match extraction envelope")
        if assertion.extraction_run_id != envelope.extraction_run_id:
            raise ValidationError("relationship extraction run must match envelope")
        if any(
            locator.filing_date != assertion.temporal.filing_date for locator in assertion.locators
        ):
            raise ValidationError("relationship locator filing dates must match temporal metadata")
        if any(locator.doc_id != envelope.document_id for locator in assertion.locators):
            raise ValidationError("relationship locators must match extraction document")
        relationships.append(
            {
                "assertion_id": assertion.assertion_id,
                "subject_entity_id": assertion.subject_entity_id,
                "object_entity_id": assertion.object_entity_id,
                "predicate": assertion.predicate,
                "source_doc_id": assertion.source_doc_id,
                "extraction_run_id": assertion.extraction_run_id,
                "status": assertion.status,
                **assertion.temporal.model_dump(mode="json"),
                "payload": assertion.model_dump_json(),
            }
        )
    if envelope.metrics and issuer_entity_id not in known:
        raise ValidationError("metrics require an explicit canonical issuer entity")
    metrics = []
    nodes = [entity_record(entity) for entity in entities]
    for metric in envelope.metrics:
        if metric.locator.filing_date != metric.temporal.filing_date:
            raise ValidationError("metric locator filing date must match temporal metadata")
        if metric.locator.doc_id != envelope.document_id:
            raise ValidationError("metric locator must match extraction document")
        metrics.append(
            {
                "observation_id": "observation:"
                + hashlib.sha256(
                    json.dumps(
                        [
                            envelope.document_id,
                            metric.metric_id,
                            metric.temporal.model_dump(mode="json"),
                        ],
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode()
                ).hexdigest(),
                "metric_id": metric.metric_id,
                "issuer_entity_id": issuer_entity_id,
                "metric_name": metric.metric_name,
                "source_doc_id": envelope.document_id,
                "extraction_run_id": envelope.extraction_run_id,
                **metric.temporal.model_dump(mode="json"),
                "payload": metric.model_dump_json(),
            }
        )
        observation_id = str(metrics[-1]["observation_id"])
        metric_entity = CanonicalEntity(
            entity_id=observation_id,
            canonical_name=metric.metric_name,
            entity_type="FinancialMetric",
            merged_surface_forms=[metric.metric_name],
            resolution_method="identifier",
            confidence=1,
        )
        nodes.append(entity_record(metric_entity))
        assertion = RelationshipAssertion(
            assertion_id="reported:" + observation_id,
            subject_entity_id=str(issuer_entity_id),
            object_entity_id=observation_id,
            predicate="REPORTED_METRIC",
            properties={
                "metric_id": metric.metric_id,
                "evidence_id": observation_id,
                "metric_observation": metric.model_dump(mode="json"),
            },
            temporal=metric.temporal,
            source_doc_id=envelope.document_id,
            extraction_run_id=envelope.extraction_run_id,
            locators=[metric.locator],
            confidence=1,
        )
        relationships.append(
            {
                "assertion_id": assertion.assertion_id,
                "subject_entity_id": assertion.subject_entity_id,
                "object_entity_id": assertion.object_entity_id,
                "predicate": assertion.predicate,
                "source_doc_id": assertion.source_doc_id,
                "extraction_run_id": assertion.extraction_run_id,
                "status": assertion.status,
                **assertion.temporal.model_dump(mode="json"),
                "payload": assertion.model_dump_json(),
            }
        )
    return nodes, relationships, metrics


@dataclass(frozen=True)
class GraphWriteResult:
    entities: int
    relationships: int
    metrics: int


class GraphWriter:
    """Write only resolved entities and source-validated, versioned assertions.

    Stores offering write_records commit the entire envelope atomically. The
    generic port fallback deliberately refuses metric writes because its base
    protocol has no metric persistence capability.
    """

    def __init__(self, store: GraphStore) -> None:
        self.store = store

    async def write_envelope(
        self,
        envelope: ExtractionEnvelope,
        entities: Sequence[CanonicalEntity],
        issuer_entity_id: str | None = None,
    ) -> GraphWriteResult:
        nodes, relationships, metrics = envelope_records(envelope, entities, issuer_entity_id)
        writer = getattr(self.store, "write_records", None)
        if writer is None:
            if metrics:
                raise ValidationError("graph store does not support atomic metric persistence")
            raise ValidationError("graph store does not support atomic envelope persistence")
        await writer(nodes, relationships, metrics)
        return GraphWriteResult(len(nodes), len(relationships), len(metrics))
