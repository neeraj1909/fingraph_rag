"""Durable ordered SSE events, separate from private workflow checkpoints."""

import json
import sqlite3
from pathlib import Path

from fingraph_rag.domain.contracts import EventEnvelope


class EventStore:
    def __init__(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute(
            "CREATE TABLE IF NOT EXISTS events (request_id TEXT NOT NULL, sequence INTEGER NOT NULL, "
            "event TEXT NOT NULL, PRIMARY KEY(request_id, sequence))"
        )
        self.connection.commit()

    def append(self, request_id, event_type, payload=None):
        with self.connection:
            row = self.connection.execute(
                "SELECT COALESCE(MAX(sequence), -1) + 1 FROM events WHERE request_id=?",
                (request_id,),
            ).fetchone()
            event = EventEnvelope(
                schema_version="1",
                request_id=request_id,
                sequence=row[0],
                event_type=event_type,
                payload=payload or {},
            )
            self.connection.execute(
                "INSERT INTO events VALUES(?,?,?)",
                (request_id, event.sequence, event.model_dump_json()),
            )
        return event

    def read(self, request_id, after=-1):
        return [
            EventEnvelope.model_validate(json.loads(row[0]))
            for row in self.connection.execute(
                "SELECT event FROM events WHERE request_id=? AND sequence>? ORDER BY sequence",
                (request_id, after),
            )
        ]

    def exists(self, request_id):
        return bool(
            self.connection.execute(
                "SELECT 1 FROM events WHERE request_id=? LIMIT 1", (request_id,)
            ).fetchone()
        )

    def close(self):
        self.connection.close()
