"""Provider-neutral extraction-agent handoff.

The core ingestion package owns the contract and validation boundary. A
PydanticAI adapter can satisfy ``ExtractionAgent`` later, but no model SDK is
required to test persistence safety or fixture replay today.
"""

from __future__ import annotations

from typing import Protocol

from fingraph_rag.domain.contracts import ExtractionEnvelope

from .contracts import CanonicalDocument, ExtractionRecord, ExtractionRun


class ExtractionAgent(Protocol):
    """Contract implemented by a local fixture or model-backed extractor."""

    def extract(self, document: CanonicalDocument, run: ExtractionRun) -> ExtractionRecord:
        """Return strictly validated extraction output for a document run."""


class FixtureExtractionAgent:
    """Replay-safe adapter for a precomputed, synthetic extraction envelope."""

    def __init__(self, envelope: ExtractionEnvelope) -> None:
        self._envelope = envelope

    def extract(self, document: CanonicalDocument, run: ExtractionRun) -> ExtractionRecord:
        envelope = self._envelope
        if envelope.document_id != document.document_id:
            raise ValueError("extraction output document does not match parsed document")
        if envelope.parser_version != run.parser_version:
            raise ValueError("extraction output parser version does not match run")
        if envelope.ontology_version != run.ontology_version:
            raise ValueError("extraction output ontology version does not match run")
        if envelope.extraction_run_id != run.run_id:
            raise ValueError("extraction output run does not match extraction run")
        return ExtractionRecord(run=run, envelope=envelope.model_copy(deep=True))
