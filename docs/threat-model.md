# Evidence and execution boundaries

Public claims pass through unverified `CandidateClaim` objects and the
`GuardrailPipeline`. Candidates cannot supply source locators: `CitationResolver`
loads retained evidence and requires an exact locator match on an asserted graph
relationship. It rejects missing, foreign, duplicate, and unrelated citations.
Graph narratives are rendered from typed relationship endpoints and predicates.

The local generator is an extractive baseline. It can quote retained evidence or
emit a canonical deterministic calculation result. It does not claim to prove
arbitrary paraphrases: number overlap and a valid graph path cannot establish
semantic entailment. Any future model adapter must emit candidates through the
same verification boundary. A model marking a claim “verified” has no authority.

Numeric verification scans actual text rather than model token declarations.
Calculations cite every input, resolve typed metric observations from graph
metadata with matching source locators, and recompute registered requests before
acceptance. Missing observations, incompatible currencies/units/restatements,
unknown formulas, bad fiscal periods, estimated inputs, nonfinite values, zero
denominators, or ambiguous input selection cause abstention. Growth formula
inputs are ordered current then prior; scale conversion uses base units, and
currency exchange is unsupported. Results retain exact request/input/version
hashes. Different rounding policies produce different hashes.

`SafeCalculator` contains a finite formula dispatch and Decimal arithmetic, with
bounded input count, magnitude, precision, and CAGR year distance. It uses no
`eval`, dynamic import, model code, subprocess, filesystem, or network operation.
This is an application capability boundary, not an operating-system sandbox.
Deployments that execute untrusted extensions require a separate restricted
worker; this code does not claim such isolation.

Input checks refuse recognizable instruction overrides, unrelated requests,
email/SSN and sensitive identifier patterns, suspected MNPI language, oversized
input, and personalized investment actions. Output checks apply the same source
instruction/sensitive-data restrictions and compliance policy. Rejected raw
claims are replaced by a fixed abstention message, so the refusal itself does not
leak the rejected content. Patterns are defense in depth and are not a complete
PII, MNPI, jailbreak, or legal classifier. Public corpus activation still requires
source terms, provenance review, and an owner-approved acquisition/privacy
policy. Legal/compliance sign-off remains a release dependency.

The retained source store is trusted for content/locator integrity. These
policies cannot prove the authenticity or truth of a maliciously substituted
source record; ingestion identity/audit and storage access controls own that
boundary. No external financial dataset is added by these tests.

Focused verification:

```bash
uv run python -m pytest tests/unit/test_calculator.py tests/unit/test_citations.py \
  tests/security/test_input_guardrails.py tests/security/test_output_guardrails.py --no-cov
```
