"""Idempotent Neo4j 5 schema, deliberately separate from query credentials."""

SCHEMA_STATEMENTS = (
    "CREATE CONSTRAINT entity_identity IF NOT EXISTS FOR (n:Entity) REQUIRE n.entity_id IS UNIQUE",
    "CREATE CONSTRAINT observation_identity IF NOT EXISTS FOR (n:MetricObservation) REQUIRE n.observation_id IS UNIQUE",
    "CREATE CONSTRAINT assertion_identity IF NOT EXISTS FOR ()-[r:ASSERTION]-() REQUIRE r.assertion_id IS UNIQUE",
    "CREATE INDEX entity_name IF NOT EXISTS FOR (n:Entity) ON (n.canonical_name)",
    "CREATE INDEX metric_period IF NOT EXISTS FOR (n:MetricObservation) ON (n.issuer_entity_id, n.fiscal_year, n.fiscal_quarter)",
    "CREATE INDEX assertion_period IF NOT EXISTS FOR ()-[r:ASSERTION]-() ON (r.fiscal_year, r.fiscal_quarter)",
)
