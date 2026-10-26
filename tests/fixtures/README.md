# Test fixtures

The ingestion fixtures in this directory are synthetic and contain no real
company, filing, or market data. They exist to test provenance, table-cell
identity, footnote links, chunking, and replay behavior without creating a
dataset licensing dependency.

The real Docling smoke test creates a deterministic PDF in a temporary
directory, labels it `synthetic-fixture`, and sends it through the same source
fetch, parser, provenance, and replay boundary. It does not download or embed
third-party financial documents.
