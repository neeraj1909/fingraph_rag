# Data lineage for the ingestion boundary

The current ingestion boundary is fixture-backed and provider-neutral. It does
not download public filings or call an LLM.

```text
SourceRegistry(source_id, source_uri, media_type, license_reference)
  -> SourceFetcher
  -> raw bytes
  -> RawArtifact(content_sha256, media_type, source_uri)
  -> DocumentParser(raw bytes when parser options select a raw adapter)
  -> CanonicalDocument(sections, tables, cells, footnotes, chunks)
  -> EntityResolver(identifier first, exact alias second)
  -> quarantine for conflicts/unresolved mentions
  -> TemporalNormalizer(issuer fiscal calendar)
  -> quarantine for invalid period labels or impossible filing dates
  -> IngestionRunReport + idempotent audit fingerprint
```

Every derived record retains a source or document identity. Table summaries may
reference a table, atomic cells, and linked footnotes simultaneously. An
unresolved entity is never inserted as a guessed graph node. An invalid period
is never silently coerced into a calendar quarter.

The synthetic fixture under `tests/fixtures/filings/` is the current CI source.
The current fixture adapter only accepts bounded `fixture://` paths. Source
ingestion may use the fetched bytes directly with a selected raw parser; the
canonical JSON sidecar remains an explicit compatibility path for the fixture
parser. Public
benchmark or SEC data must be added later with a dataset manifest, license/terms
record, source hash, and replay policy.
