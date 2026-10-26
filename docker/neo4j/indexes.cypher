CREATE INDEX entity_name IF NOT EXISTS FOR (n:Entity) ON (n.canonical_name);
CREATE INDEX metric_period IF NOT EXISTS FOR (n:MetricObservation) ON (n.issuer_entity_id, n.fiscal_year, n.fiscal_quarter);
CREATE INDEX assertion_period IF NOT EXISTS FOR ()-[r:ASSERTION]-() ON (r.fiscal_year, r.fiscal_quarter);
