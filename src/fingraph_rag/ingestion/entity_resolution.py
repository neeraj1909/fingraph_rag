"""Identifier-first entity resolution with explicit quarantine outcomes."""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Iterable
from datetime import UTC, datetime

from pydantic import Field

from fingraph_rag.domain.contracts import CanonicalEntity, ContractModel, EntityMention

from .contracts import EntityResolutionBatch, QuarantineRecord
from .identity import content_sha256, stable_id


def normalize_surface_form(value: str) -> str:
    """Normalize aliases for exact comparison without changing their meaning."""

    return " ".join(re.sub(r"[^a-z0-9]+", " ", value.casefold()).split())


def normalize_identifier(key: str, value: str) -> tuple[str, str]:
    """Normalize identifier keys and values for deterministic lookup."""

    normalized_key = re.sub(r"[^a-z0-9]+", "", key.casefold())
    normalized_value = re.sub(r"[^a-z0-9]+", "", value.casefold())
    return normalized_key, normalized_value


class EntityResolutionOutcome(ContractModel):
    batch: EntityResolutionBatch
    quarantined: list[QuarantineRecord] = Field(default_factory=list)


class EntityIndex:
    """Read-only canonical entity index used by the resolver."""

    def __init__(self, entities: Iterable[CanonicalEntity]) -> None:
        self._entities = tuple(entities)
        self._by_identifier: dict[tuple[str, str], list[CanonicalEntity]] = defaultdict(list)
        self._by_alias: dict[str, list[CanonicalEntity]] = defaultdict(list)
        for entity in self._entities:
            for key, value in entity.identifiers.items():
                self._by_identifier[normalize_identifier(key, value)].append(entity)
            aliases = [entity.canonical_name, *entity.merged_surface_forms]
            for alias in aliases:
                self._by_alias[normalize_surface_form(alias)].append(entity)

    def identifier_candidates(self, mention: EntityMention) -> list[CanonicalEntity]:
        candidates: dict[str, CanonicalEntity] = {}
        for key, value in mention.identifiers.items():
            for entity in self._by_identifier.get(normalize_identifier(key, value), []):
                candidates[entity.entity_id] = entity
        return list(candidates.values())

    def alias_candidates(self, mention: EntityMention) -> list[CanonicalEntity]:
        candidates = self._by_alias.get(normalize_surface_form(mention.surface_form), [])
        return list({entity.entity_id: entity for entity in candidates}.values())


class EntityResolver:
    """Resolve high-confidence identifiers, then exact aliases, else quarantine."""

    def __init__(self, index: EntityIndex, *, policy_version: str = "identifier-first-v1") -> None:
        self._index = index
        self._policy_version = policy_version

    @property
    def policy_version(self) -> str:
        return self._policy_version

    def resolve(
        self,
        *,
        document_id: str,
        run_id: str,
        mentions: Iterable[EntityMention],
        now: datetime | None = None,
    ) -> EntityResolutionOutcome:
        timestamp = now or datetime.now(UTC)
        decisions = []
        quarantined = []
        unresolved = []

        for mention in mentions:
            identifier_candidates = self._index.identifier_candidates(mention)
            alias_candidates = self._index.alias_candidates(mention)
            candidates = identifier_candidates or alias_candidates

            if len(identifier_candidates) == 1:
                selected = identifier_candidates[0]
                decision = "merge"
                reason_code = "identifier_match"
                confidence = 1.0
            elif len(identifier_candidates) > 1:
                selected = None
                decision = "defer"
                reason_code = "identifier_conflict"
                confidence = 0.0
            elif len(alias_candidates) == 1:
                selected = alias_candidates[0]
                decision = "merge"
                reason_code = "exact_alias_match"
                confidence = 0.98
            elif len(alias_candidates) > 1:
                selected = None
                decision = "defer"
                reason_code = "alias_conflict"
                confidence = 0.0
            else:
                selected = None
                decision = "defer"
                reason_code = "no_identifier_or_alias_match"
                confidence = 0.0

            resolution = self._decision(
                mention=mention,
                candidates=candidates,
                selected=selected,
                decision=decision,
                reason_code=reason_code,
                confidence=confidence,
            )
            decisions.append(resolution)

            if decision == "defer":
                unresolved.append(mention.surface_form)
                quarantined.append(
                    QuarantineRecord(
                        quarantine_id=stable_id(
                            "quarantine", (run_id, "entity_resolution", mention.surface_form)
                        ),
                        document_id=document_id,
                        run_id=run_id,
                        stage="entity_resolution",
                        code=reason_code,
                        message="Entity requires review before graph insertion.",
                        subject=mention.surface_form,
                        payload_hash=content_sha256(mention.model_dump_json().encode("utf-8")),
                        created_at=timestamp,
                    )
                )

        return EntityResolutionOutcome(
            batch=EntityResolutionBatch(
                run_id=run_id,
                document_id=document_id,
                decisions=decisions,
                unresolved_surface_forms=unresolved,
                policy_version=self._policy_version,
            ),
            quarantined=quarantined,
        )

    @staticmethod
    def _decision(
        *,
        mention: EntityMention,
        candidates: list[CanonicalEntity],
        selected: CanonicalEntity | None,
        decision: str,
        reason_code: str,
        confidence: float,
    ):
        from fingraph_rag.domain.contracts import EntityResolutionDecision

        return EntityResolutionDecision(
            mention=mention,
            candidates=candidates,
            selected_entity_id=selected.entity_id if selected else None,
            decision=decision,
            reason_code=reason_code,
            evidence=[mention.locator],
            confidence=confidence,
        )
