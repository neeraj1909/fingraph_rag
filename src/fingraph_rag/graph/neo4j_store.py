"""Neo4j 5 transactional HTTP adapter with atomic, idempotent envelope writes.

Uses only the standard library. The query-facing adapter accepts template IDs,
never raw Cypher. Production deployments should supply separate read-only
credentials; the local Community runtime does not provide granular roles.
"""

import asyncio
import base64
import hashlib
import json
from collections.abc import Mapping, Sequence
from datetime import date
from typing import Any, get_args
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

from fingraph_rag.domain.contracts import (
    CanonicalEntity,
    CypherQueryPlan,
    EntityType,
    RelationshipAssertion,
)
from fingraph_rag.domain.errors import DependencyUnavailable, PolicyViolation, ValidationError
from fingraph_rag.graph.cypher_ast_policy import CypherAstPolicy
from fingraph_rag.graph.cypher_templates import MAX_RESULTS, resolve_template
from fingraph_rag.graph.migrations import SCHEMA_STATEMENTS

ENTITY_TYPES = frozenset(get_args(EntityType))
_LABEL_WRITES = "\n".join(
    f"FOREACH (typed IN CASE WHEN row.entity_type = '{label}' THEN [1] ELSE [] END | SET n:{label})"
    for label in sorted(ENTITY_TYPES)
)
ENTITY_WRITE = (
    """UNWIND $records AS row
MERGE (n:Entity {entity_id: row.entity_id}) SET n += row
"""
    + _LABEL_WRITES
    + "\nRETURN count(n) AS count"
)
RELATIONSHIP_WRITE = """UNWIND $records AS row
MATCH (a:Entity {entity_id: row.subject_entity_id})
MATCH (b:Entity {entity_id: row.object_entity_id})
MERGE (a)-[r:ASSERTION {assertion_id: row.assertion_id}]->(b)
ON CREATE SET r += row
FOREACH (conflict IN CASE WHEN r.payload <> row.payload THEN [1] ELSE [] END |
CREATE (a)-[:ASSERTION {assertion_id: row.assertion_id}]->(b))
RETURN count(r) AS count"""
METRIC_WRITE = """UNWIND $records AS row
MERGE (n:MetricObservation {observation_id: row.observation_id})
ON CREATE SET n += row
FOREACH (conflict IN CASE WHEN n.payload <> row.payload THEN [1] ELSE [] END |
CREATE (:MetricObservation {observation_id: row.observation_id}))
RETURN count(n) AS count"""


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise DependencyUnavailable("Neo4j redirects are not permitted")


class Neo4jGraphStore:
    def __init__(
        self,
        url: str = "http://localhost:7474",
        *,
        username: str = "neo4j",
        password: str,
        database: str = "neo4j",
        read_username: str | None = None,
        read_password: str | None = None,
        timeout_seconds: float = 30,
    ) -> None:
        parsed = urlsplit(url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc or parsed.username:
            raise ValueError("Neo4j URL must be an HTTP(S) origin without credentials")
        if parsed.path not in {"", "/"} or parsed.query or parsed.fragment:
            raise ValueError("Neo4j URL must not include a path, query, or fragment")
        if not 0 < timeout_seconds <= 30:
            raise ValueError("Neo4j timeout must be between 0 and 30 seconds")
        if bool(read_username) != bool(read_password):
            raise ValueError("both read username and password must be supplied")
        self.endpoint = f"{url.rstrip('/')}/db/{quote(database, safe='')}/tx/commit"
        self.timeout_seconds = timeout_seconds
        self._write_auth = self._auth(username, password)
        self._read_auth = self._auth(read_username or username, read_password or password)
        self.policy = CypherAstPolicy()

    @staticmethod
    def _auth(username: str, password: str) -> str:
        return "Basic " + base64.b64encode(f"{username}:{password}".encode()).decode()

    def _transaction(
        self, statements: Sequence[tuple[str, Mapping[str, Any]]], *, read: bool = False
    ) -> list[list[dict[str, Any]]]:
        payload = json.dumps(
            {
                "statements": [
                    {"statement": query, "parameters": dict(params), "resultDataContents": ["row"]}
                    for query, params in statements
                ]
            }
        ).encode()
        request = Request(
            self.endpoint,
            data=payload,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "Authorization": self._read_auth if read else self._write_auth,
                "max-execution-time": str(int(self.timeout_seconds * 1000)),
            },
        )
        try:
            with build_opener(_NoRedirect()).open(
                request, timeout=self.timeout_seconds
            ) as response:
                raw = response.read(8_000_001)
            if len(raw) > 8_000_000:
                raise DependencyUnavailable("Neo4j response exceeds byte budget")
            result = json.loads(raw)
        except (HTTPError, URLError, TimeoutError, OSError, ValueError) as exc:
            raise DependencyUnavailable("Neo4j transaction failed") from exc
        if result.get("errors"):
            # Do not expose server messages, query text, or credentials to callers.
            code = result["errors"][0].get("code", "unknown")
            raise DependencyUnavailable(f"Neo4j rejected transaction ({code})")
        return [
            [dict(zip(item["columns"], row["row"], strict=True)) for row in item["data"]]
            for item in result["results"]
        ]

    async def initialize_schema(self) -> None:
        # Neo4j prohibits schema and data changes in the same transaction.
        for statement in SCHEMA_STATEMENTS:
            await asyncio.to_thread(self._transaction, [(statement, {})])

    async def execute_read(
        self,
        template_id: str | CypherQueryPlan,
        parameters: Mapping[str, Any] | None = None,
    ) -> Sequence[Mapping[str, Any]]:
        if isinstance(template_id, CypherQueryPlan):
            if parameters is not None:
                raise ValidationError("plan parameters must be supplied only once")
            parameters = template_id.parameter_values
            plan_hops = template_id.max_hops
            template_id = template_id.template_id
        else:
            plan_hops = None
        params = dict(parameters or {})
        allowed = {
            "entity_ids",
            "max_hops",
            "limit",
            "fiscal_year",
            "fiscal_quarter",
            "restatement_version",
            "period_start",
            "period_end",
            "as_of_date",
            "status",
        }
        if set(params) - allowed:
            raise PolicyViolation("unknown graph query parameter")
        ids = params.get("entity_ids", [])
        if (
            not isinstance(ids, list)
            or not 1 <= len(ids) <= 100
            or any(not isinstance(item, str) or not item or len(item) > 256 for item in ids)
        ):
            raise ValidationError("entity_ids requires 1..100 bounded canonical IDs")
        limit = params.get("limit", 20)
        if type(limit) is not int or not 1 <= limit <= MAX_RESULTS:
            raise PolicyViolation("graph result limit must be an integer between 1 and 100")
        hops = params.pop("max_hops", plan_hops if plan_hops is not None else 1)
        if template_id in {"entity_lookup", "metric_observations"} and hops == 0:
            hops = 1
        if type(hops) is not int or not 1 <= hops <= 4:
            raise PolicyViolation("graph traversal requires 1..4 hops")
        if (
            plan_hops is not None
            and template_id not in {"entity_lookup", "metric_observations"}
            and hops > plan_hops
        ):
            raise PolicyViolation("template parameters exceed the query plan hop budget")
        year = params.get("fiscal_year")
        if year is not None and (type(year) is not int or not 1900 <= year <= 2200):
            raise ValidationError("invalid graph fiscal year")
        quarter = params.get("fiscal_quarter")
        if quarter is not None and quarter not in {"Q1", "Q2", "Q3", "Q4", "FY"}:
            raise ValidationError("invalid graph fiscal quarter")
        version = params.get("restatement_version")
        if version is not None and (type(version) is not int or version < 0):
            raise ValidationError("invalid graph restatement version")
        for field in ("period_start", "period_end"):
            if params.get(field) is not None:
                try:
                    params[field] = date.fromisoformat(str(params[field])).isoformat()
                except ValueError as exc:
                    raise ValidationError(f"{field} must be an ISO date") from exc
        if (
            params.get("period_start") is not None
            and params.get("period_end") is not None
            and params["period_end"] < params["period_start"]
        ):
            raise ValidationError("period_end must be on or after period_start")
        if params.get("status", "asserted") != "asserted":
            raise PolicyViolation("only asserted graph evidence may be retrieved")
        if params.get("as_of_date") is not None:
            try:
                params["as_of_date"] = date.fromisoformat(str(params["as_of_date"])).isoformat()
            except ValueError as exc:
                raise ValidationError("as_of_date must be an ISO date") from exc
        try:
            query = resolve_template(template_id, max_hops=hops)
        except ValueError as exc:
            raise PolicyViolation(str(exc)) from exc
        parsed = self.policy.validate(query)
        defaults: dict[str, Any] = {
            "entity_ids": ids,
            "limit": limit,
            "fiscal_year": None,
            "fiscal_quarter": None,
            "restatement_version": None,
            "period_start": None,
            "period_end": None,
            "as_of_date": None,
            "status": "asserted",
        }
        defaults.update(params)
        bound = {name: defaults[name] for name in parsed.parameters}
        rows = (await asyncio.to_thread(self._transaction, [(query, bound)], read=True))[0]
        if len(rows) > limit:
            raise PolicyViolation("Neo4j exceeded graph result budget")
        decoded = []
        for row in rows:
            if "relationships" in row:
                row["relationships"] = [json.loads(value) for value in row["relationships"]]
                digest = hashlib.sha256(
                    json.dumps(
                        [row["entity_ids"], row["relationship_ids"]], separators=(",", ":")
                    ).encode()
                ).hexdigest()[:24]
                row["path_id"] = "path:" + digest
            if "payload" in row:
                row["payload"] = json.loads(row["payload"])
            decoded.append(row)
        return decoded

    async def upsert_entities(self, entities: Sequence[Mapping[str, Any]]) -> int:
        if not entities:
            return 0
        self._validate_entities(entities)
        rows = await asyncio.to_thread(
            self._transaction, [(ENTITY_WRITE, {"records": list(entities)})]
        )
        return rows[0][0]["count"]

    async def upsert_relationships(self, relationships: Sequence[Mapping[str, Any]]) -> int:
        if not relationships:
            return 0
        self._validate_relationships(relationships)
        endpoint_ids = list(
            {
                str(record[key])
                for record in relationships
                for key in ("subject_entity_id", "object_entity_id")
            }
        )
        existing = await self.execute_read(
            "entity_lookup", {"entity_ids": endpoint_ids, "limit": 100}
        )
        if len(existing) != len(endpoint_ids):
            raise ValidationError("relationship endpoints are missing")
        rows = await asyncio.to_thread(
            self._transaction, [(RELATIONSHIP_WRITE, {"records": list(relationships)})]
        )
        count = rows[0][0]["count"]
        if count != len(relationships):
            raise ValidationError("relationship endpoints are missing")
        return count

    async def write_records(
        self,
        entities: Sequence[Mapping[str, Any]],
        relationships: Sequence[Mapping[str, Any]],
        metrics: Sequence[Mapping[str, Any]],
    ) -> None:
        self._validate_entities(entities)
        self._validate_relationships(relationships)
        # Reject unresolved edges before sending any mutation to the server.
        known = {record.get("entity_id") for record in entities}
        if any(not record.get("entity_id") for record in entities):
            raise ValidationError("entity record requires entity_id")
        for record in relationships:
            if (
                record.get("subject_entity_id") not in known
                or record.get("object_entity_id") not in known
            ):
                raise ValidationError("all relationship endpoints must be included in atomic write")
        # All source facts and endpoints commit or roll back as one server transaction.
        await asyncio.to_thread(
            self._transaction,
            [
                (ENTITY_WRITE, {"records": list(entities)}),
                (RELATIONSHIP_WRITE, {"records": list(relationships)}),
                (METRIC_WRITE, {"records": list(metrics)}),
            ],
        )

    @staticmethod
    def _validate_entities(entities: Sequence[Mapping[str, Any]]) -> None:
        for record in entities:
            try:
                entity = CanonicalEntity.model_validate_json(record["payload"])
                if any(
                    record.get(key) != getattr(entity, key)
                    for key in ("entity_id", "canonical_name", "entity_type")
                ):
                    raise ValueError("entity fields differ from payload")
            except (KeyError, ValueError, TypeError) as exc:
                raise ValidationError("invalid canonical graph entity record") from exc

    @staticmethod
    def _validate_relationships(relationships: Sequence[Mapping[str, Any]]) -> None:
        for record in relationships:
            try:
                assertion = RelationshipAssertion.model_validate_json(record["payload"])
                expected = {
                    key: getattr(assertion, key)
                    for key in (
                        "assertion_id",
                        "subject_entity_id",
                        "object_entity_id",
                        "predicate",
                        "source_doc_id",
                        "extraction_run_id",
                        "status",
                    )
                }
                expected.update(assertion.temporal.model_dump(mode="json"))
                if any(record.get(key) != value for key, value in expected.items()):
                    raise ValueError("assertion fields differ from payload")
                if any(
                    locator.doc_id != assertion.source_doc_id
                    or locator.filing_date != assertion.temporal.filing_date
                    for locator in assertion.locators
                ):
                    raise ValueError("assertion locator metadata differs from source")
            except (KeyError, ValueError, TypeError) as exc:
                raise ValidationError("invalid source-backed graph assertion record") from exc
