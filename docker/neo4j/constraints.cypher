CREATE CONSTRAINT entity_identity IF NOT EXISTS FOR (n:Entity) REQUIRE n.entity_id IS UNIQUE;
CREATE CONSTRAINT observation_identity IF NOT EXISTS FOR (n:MetricObservation) REQUIRE n.observation_id IS UNIQUE;
CREATE CONSTRAINT assertion_identity IF NOT EXISTS FOR ()-[r:ASSERTION]-() REQUIRE r.assertion_id IS UNIQUE;
