# Reproducible evaluation

Run `uv run python scripts/run_eval.py --dataset evals/datasets/smoke.jsonl`.
The scored smoke dataset and corpus are repository-owned synthetic data. They
test retrieval wiring, source identity, path coverage, and temporal constraints;
they are not an accuracy benchmark for real financial questions.

## Public source datasets

FinQA and DocFinQA are available for local research and validation as their
original JSON files. Their immutable revisions, split URLs, expected byte
sizes, source hashes, row counts, formats, and source-license statements are
recorded in [`public_sources.json`](datasets/public_sources.json). The raw
files are stored under `evals/data/public/`, which is gitignored; no 4.7 GB
DocFinQA corpus is copied into the repository or transformed into synthetic
gold labels.

Download and audit all seven pinned files:

```bash
uv run python scripts/fetch_public_datasets.py \
  --manifest evals/datasets/public_sources.json \
  --output evals/data/public
```

After downloading, repeat the complete hash, split-count, and row-format audit
without network access:

```bash
uv run python scripts/fetch_public_datasets.py \
  --manifest evals/datasets/public_sources.json \
  --output evals/data/public \
  --check-only
```

The audit checks every row. FinQA uses nested `qa.question`, `qa.answer`, and
`qa.program` fields plus table/text context. DocFinQA records are flat
`Context`, `Question`, `Program`, and `Answer` objects. FinQA `private_test`
has questions but no answer/program labels and is counted as unlabeled; it is
not scored as a zero. These source formats do not match the project's scored
JSONL schema, so `scripts/run_eval.py` continues to score the explicitly labeled
synthetic smoke set only. No live RAG accuracy claim is made from these raw
downloads.

FinQA's project site describes its dataset as CC BY 4.0 while its repository
license is MIT; the source manifest records those separately. The DocFinQA
dataset card advertises MIT, but this is not a legal conclusion about embedded
SEC filing text. Confirm source and redistribution terms before external use.

The plan's financial `SEC-QA` name refers to a refreshable generation framework,
not a verified fixed public split download. The public `SecQA` benchmark is a
different computer-security QA dataset and is intentionally not substituted.
The project-specific multi-hop SEC and adversarial collections remain
repository-owned fixtures rather than public dataset downloads.

The fetcher uses immutable revision URLs, streaming SHA-256 verification,
atomic local files, and a standard-library row audit. Hugging Face documents
dataset revision history and JSON/JSONL file formats in its [dataset sharing
guide](https://huggingface.co/docs/datasets/en/share/) and [loading
guide](https://huggingface.co/docs/datasets/en/loading/). No dataset SDK or
runtime dependency is added.

### Implementation rationale

- The task is reproducible acquisition and source-format verification, not
  converting public examples into the project's synthetic QA schema. Keeping
  upstream JSON intact preserves source fields and avoids inventing graph-path
  or adversarial gold labels.
- `scripts/fetch_public_datasets.py` streams bytes and validates records
  incrementally; it keeps memory bounded to the current JSON row. A pinned
  SHA-256 protects both first download and later `--check-only` runs. Temporary
  files plus no-clobber publication protect an existing cache from partial or
  corrupt replacements.
- The official Hugging Face Datasets docs support dataset revision history,
  commit references, JSON and JSONL sources. This project uses direct pinned
  file URLs and Python's standard library, so the `datasets` SDK version is
  inapplicable and no new loader dependency is required.
- Focused behavior evidence:
  `timeout 60 uv run python -m pytest -q tests/unit/test_public_dataset_fetch.py --no-cov`
  and `uv run ruff check scripts/fetch_public_datasets.py tests/unit/test_public_dataset_fetch.py`.

The CLI replays the same corpus through the production `RetrievalService` with
a graph-free baseline and graph-enabled system. Its manifest explicitly labels
`fixture-replay-no-embedding`: no claim of live dense/sparse/provider quality is
made for this replay. Inject actual storage ports into `NaiveHybridBaseline` and
`GraphRagSystem` for a live benchmark. The live Qdrant application integration
check is `FINGRAPH_RUN_STORAGE_INTEGRATION=1 uv run pytest
tests/integration/test_retrieval_live.py --no-cov`.

Every run creates a new immutable directory with a shared manifest and report.
The manifest hashes dataset, corpus, source tree, dependencies, retrieval
budgets, software commit, and environment. Repeated runs preserve manifest
identity and evidence metrics; measured latency may vary.

All specified metrics appear in the report with measured-case counts. Missing
gold data, provider cost, Cypher execution evidence, or generation is `null`,
not a passing zero. The default CLI uses the deterministic `extractive` runner:
it measures `faithfulness`, `numerical_em`, `citation_precision`,
`citation_recall`, `temporal_consistency`, `abstention_rate`, and the
accepted-to-completion `p50_latency_ms`/`p95_latency_ms` when labeled claims are
available. Its numerical, citation, and temporal scores use explicit synthetic
gold fields in each JSONL case and verified extractive claims only. It does not
claim provider cost or Cypher success. `run_ablation(..., generation=runner)`
accepts an actual application generation boundary; only finite, verifier-owned
values for those metrics are included. Without that callback, all generation
metrics and accepted-to-completion latency remain explicitly unmeasured.
Retrieval latency is reported separately for every run. Synthetic path and
evidence labels are embedded in each JSONL case.

For safe telemetry configure `configure_otel(output_path=Path(...),
endpoint="http://localhost:6006/v1/traces")` and install it with `set_telemetry`.
The endpoint uses OTLP HTTP and can target a local Phoenix collector. No raw
questions, document content, prompt text, exceptions, or SDK span events are
exported. Verify local evidence with `uv run python scripts/export_trace.py
--path reports/traces/trace.jsonl --verify-redaction`.
