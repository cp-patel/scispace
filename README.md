# Research Assistant

Give it a research topic; it picks the ~40 most relevant papers out of thousands using hand-written ranking math (BM25F, embeddings, Personalized PageRank, rank fusion, Bayes-calibrated screening, submodular and DPP selection), then a LangGraph agent writes a review where every claim cites a quote that was checked word for word against the source.

![Architecture](docs/architecture-rag.svg)

## Quick start

You need Docker. The default "offline" brain needs no API keys.

```bash
cp .env.example .env
```

```bash
docker compose up -d --build
```

```bash
curl -s -X POST localhost:8000/v1/jobs -H "X-API-Key: $(grep '^API_KEY=' .env | cut -d= -f2)" -H 'Content-Type: application/json' -d '{"topic":"retrieval-augmented generation for scientific question answering"}'
```

The last command returns a job id. Check progress with `GET /v1/jobs/<id>` and, once `status` is `succeeded`, fetch the paper with `GET /v1/jobs/<id>/paper` (both need the same `X-API-Key` header). Interactive API docs are at http://localhost:8000/docs. Set a long random `API_KEY` in `.env` before exposing the server anywhere.

## Sample outputs

From a real run (offline brain, live arXiv, OpenAlex answers from the HTTP cache of an earlier live run; about 80 seconds, or about 3 minutes with a cold cache):

- [samples/paper.md](samples/paper.md): the review
- [samples/references.bib](samples/references.bib): references built from metadata
- [samples/report.md](samples/report.md): the funnel (373 candidates, 150 shortlisted, 40 selected), every pick and its probability, section checks, time per step
- [samples/selection.json](samples/selection.json) and [samples/trace.json](samples/trace.json)

## Switching on Claude

Set `ANTHROPIC_API_KEY` and `BRAIN=claude` in `.env` (or pass `"brain": "claude"` per job), then restart. Every call's tokens and cost are logged against the job; a job stops once it passes its budget (`JOB_BUDGET_USD`, default $10, or `budget_usd` per job). Models and effort levels can be set per step (`MODEL_SCREEN`, `EFFORT_WRITE_SECTION`, and so on); the default model is `claude-opus-5`.

A free OpenAlex key (`OPENALEX_API_KEY`) raises the daily OpenAlex allowance from $0.10 to $1, which the evals and the benchmark need.

## Tests, evals, and the vector lab

```bash
uv sync --extra lab
```

```bash
uv run pytest
```

```bash
docker compose up -d postgres && uv run pytest -m db
```

- `uv run pytest -m slow` runs the larger HNSW recall check; `uv run ruff check src tests lab evals` lints.
- `uv run python -m evals.selection_eval` scores selection against 8 published surveys (results in `evals/results/`); `uv run python -m evals.writing_eval` checks finished jobs.
- `uv run python -m lab.bench --source synthetic --n 5000 --queries 200` benchmarks LSH, IVF, PQ and HNSW; `--source openalex --n 50000 --pgvector` runs it on real abstracts and pgvector.

Set `HTTP_CACHE_DIR=.http-cache` so repeated OpenAlex calls are free.

## Layout

New to the code? Start with [docs/code-tour.md](docs/code-tour.md): a reading order, one job traced through the code, a map of every file, and where to make common changes.

| Path | What it holds |
|---|---|
| `src/research_assistant/ranking/` | the math: BM25F, Rocchio, PageRank, RRF, Bayes, selection, DPP, spectral clustering, metrics |
| `src/research_assistant/funnel.py`, `writing.py` | paper selection and evidence-based writing (no database, so tests and evals share them) |
| `src/research_assistant/agent/`, `worker.py` | the LangGraph graph and the crash-safe worker |
| `src/research_assistant/api/`, `db/` | the HTTP API and the Postgres schema, queue and results |
| `src/research_assistant/brain/` | the offline brain and the Claude brain |
| `lab/`, `evals/`, `tests/` | vector search lab, evals, tests |

## Documentation

- [docs/code-tour.md](docs/code-tour.md): how to start and find your way around the code
- [docs/interview-prep.md](docs/interview-prep.md): the one-stop interview prep: what to say, the math, results, stories, weak spots, and a 45-question drill
- [docs/architecture.md](docs/architecture.md): how it works, why, and how it grows to 200 million papers
- [docs/math.md](docs/math.md): every algorithm, with proofs, tests, and our measured results
- [docs/study-guide.md](docs/study-guide.md): a five-day plan, a traced run, and 45 interview questions with answers
