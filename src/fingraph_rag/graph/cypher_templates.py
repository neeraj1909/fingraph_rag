"""Closed, parameterized read templates; user text never becomes Cypher."""

MAX_RESULTS = 100

ENTITY_LOOKUP = """MATCH (n:Entity)
WHERE n.entity_id IN $entity_ids
RETURN n.payload AS payload
ORDER BY n.entity_id LIMIT $limit"""
METRIC_OBSERVATIONS = """MATCH (n:MetricObservation)
WHERE n.issuer_entity_id IN $entity_ids
AND ($fiscal_year IS NULL OR n.fiscal_year = $fiscal_year)
AND ($fiscal_quarter IS NULL OR n.fiscal_quarter = $fiscal_quarter)
AND ($restatement_version IS NULL OR n.restatement_version = $restatement_version)
AND ($period_start IS NULL OR (n.period_start IS NOT NULL AND n.period_start >= $period_start))
AND ($period_end IS NULL OR (n.period_end IS NOT NULL AND n.period_end <= $period_end))
AND ($as_of_date IS NULL OR (n.filing_date <= $as_of_date
    AND (n.valid_from IS NULL OR n.valid_from <= $as_of_date)
    AND (n.valid_to IS NULL OR n.valid_to >= $as_of_date)))
RETURN n.payload AS payload ORDER BY n.metric_id LIMIT $limit"""


def path_template(max_hops: int) -> str:
    if type(max_hops) is not int or not 1 <= max_hops <= 4:
        raise ValueError("max_hops must be an integer between 1 and 4")
    return f"""MATCH p=(a:Entity)-[:ASSERTION*1..{max_hops}]-(b:Entity)
WHERE a.entity_id IN $entity_ids
AND all(r IN relationships(p) WHERE r.status = $status
AND ($fiscal_year IS NULL OR r.fiscal_year = $fiscal_year)
AND ($fiscal_quarter IS NULL OR r.fiscal_quarter = $fiscal_quarter)
AND ($restatement_version IS NULL OR r.restatement_version = $restatement_version)
AND ($period_start IS NULL OR (r.period_start IS NOT NULL AND r.period_start >= $period_start))
AND ($period_end IS NULL OR (r.period_end IS NOT NULL AND r.period_end <= $period_end))
AND ($as_of_date IS NULL OR (r.filing_date <= $as_of_date
    AND (r.valid_from IS NULL OR r.valid_from <= $as_of_date)
    AND (r.valid_to IS NULL OR r.valid_to >= $as_of_date))) )
RETURN [n IN nodes(p) | n.entity_id] AS entity_ids,
[r IN relationships(p) | r.assertion_id] AS relationship_ids,
[r IN relationships(p) | r.payload] AS relationships,
length(p) AS hop_count
ORDER BY hop_count, relationship_ids LIMIT $limit"""


PATH_TEMPLATE_IDS = frozenset({"entity_facts", "bounded_paths", "temporal_facts", "thematic_paths"})
TRUSTED_READ_QUERIES = (
    ENTITY_LOOKUP,
    METRIC_OBSERVATIONS,
    *(path_template(h) for h in range(1, 5)),
)


def resolve_template(template_id: str, *, max_hops: int = 1) -> str:
    if template_id == "entity_lookup":
        return ENTITY_LOOKUP
    if template_id == "metric_observations":
        return METRIC_OBSERVATIONS
    if template_id in PATH_TEMPLATE_IDS:
        return path_template(max_hops)
    raise ValueError(f"unknown graph read template: {template_id}")
