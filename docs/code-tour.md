# Code tour

A guided way into the code. Start at the top and stop whenever you have what you need. All paths under `src/` are inside `src/research_assistant/`.

## 1. Get it running (10 minutes)

You need [uv](https://docs.astral.sh/uv/) and Docker.

```bash
uv sync --extra lab
```

```bash
uv run pytest
```

The fast suite needs nothing else and finishes in a few seconds. Seeing it pass tells you the setup works. Next, start the database and run the database tests:

```bash
docker compose up -d postgres && uv run pytest -m db
```

To run the whole service and start a real job, follow the Quick start in the [README](../README.md). The interactive API docs are at http://localhost:8000/docs once it is up.

## 2. The one idea that explains the layout

The code keeps **pure logic** apart from **input and output** (I/O):

- **Pure logic** takes data in and returns data out. It never touches the database or the network, so tests and evals can call it directly.
  - `funnel.py` picks the papers.
  - `writing.py` builds and checks the paper.
  - `ranking/` holds all the math.
- **I/O wrappers** load data, call the pure logic, and save results.
  - `agent/nodes.py` does this for each step of the graph.
  - `db/`, `sources/` and `api/` are the other I/O layers.

The model is behind one interface, the **Brain** (`brain/base.py`). Every step that needs judgment or writing calls it. There are two implementations:
- `OfflineBrain`: rules, free.
- `ClaudeBrain`: the Anthropic API.

So the rest of the code never knows which model is running.

A useful picture is a kitchen. `funnel.py` and `writing.py` are the recipes, `agent/nodes.py` is the cook who fetches ingredients and plates the dish, and the Brain is the taste tester you can swap.

## 3. Reading order

Read these in order. Each stop builds on the one before.

| # | File | What to look for | Time |
|---|---|---|---|
| 1 | `README.md`, `docs/architecture-rag.svg` | What the system does and the picture of it | 5 min |
| 2 | `src/config.py` | `AlgoParams`: every knob (BM25 weights, K = 40, budgets) in one place; `JobSettings`: what a job request can contain | 5 min |
| 3 | `src/agent/graph.py` | The whole workflow in 25 lines: the steps, the section loop, the fan-out | 5 min |
| 4 | `src/funnel.py` | `gather` → `compute_signals` → `shortlist` → `screen` → `select`: paper selection end to end | 20 min |
| 5 | `src/ranking/` | The math behind step 4, one small file per method; read alongside [math.md](math.md) | 30 min |
| 6 | `src/writing.py` | `rank_chunks`, `pick_within_budget`, `accept_cards`, `verify`, `finalize`, `assemble_paper` | 20 min |
| 7 | `src/agent/nodes.py` | The `@step` wrapper, then each node: how pure logic gets its inputs and saves its outputs | 20 min |
| 8 | `src/worker.py`, `src/db/jobs.py` | How jobs are claimed, kept alive, resumed after a crash, and finished | 15 min |
| 9 | `src/brain/base.py`, `offline.py`, `claude.py` | The Brain interface and its two implementations | 15 min |
| 10 | `src/api/app.py` | The HTTP routes; they only read and write rows | 10 min |

## 4. Follow one job through the code

This is the path a single request takes. Open each file as you go.

```
POST /v1/jobs                          api/app.py: create()
  ↓ saves the job as "queued"          db/jobs.py: create_job()
Worker loop                            worker.py: Worker.run_forever()
  ↓ claims it (SKIP LOCKED)            db/jobs.py: claim_job()
  ↓ starts a heartbeat                 worker.py: Worker._heartbeat() → db/jobs.py: renew_lease()
  ↓ runs the graph                     worker.py: Worker.run_job() → graph.ainvoke()
Graph steps                            agent/graph.py: build_graph()
  plan      → nodes.plan()       → brain.plan()                      topic → facets
  gather    → nodes.gather()     → funnel.gather()                   OpenAlex search + snowballing
  rank      → nodes.rank()       → funnel.compute_signals()          BM25F, dense, PageRank, RRF
  screen    → nodes.screen()     → funnel.screen()                   brain verdicts + Bayes
  select    → nodes.select()     → funnel.select()                   submodular / MMR / DPP / top-k
  approve   → nodes.approve()    → interrupt()                       optional human pause
  ingest    → nodes.ingest()     → sources/arxiv.py, text/           PDF → clean text → chunks
  outline   → nodes.outline()    → ranking/cluster.py                spectral clustering → sections
  section (one branch per section, in parallel via Send):
    retrieve → writing.rank_chunks(), pick_within_budget()
    compress → brain.compress() → writing.accept_cards()             evidence cards, quotes checked
    write    → brain.write_section()
    verify   → writing.verify()  (loops to revise up to 2 times)
    finish   → writing.finalize()
  assemble  → nodes.assemble()   → writing.assemble_paper()          references from metadata
  ↓ marks the job "succeeded"          db/jobs.py: finish_job()
GET /v1/jobs/{id}/paper                api/app.py: paper()
```

Every step runs inside the `@step` wrapper in `agent/nodes.py`. Before the step runs, the wrapper:
- stops if the job was canceled, the lease was lost, or the worker is shutting down;
- tags logs and LLM costs with the job and step.

After the step, it records the step's time and counts; that record is the per-job trace. LangGraph saves a checkpoint after each step. That's why a crashed job can resume on another worker.

## 5. File map

**Service**

| File | What it does |
|---|---|
| `src/config.py` | All settings, from environment variables or `.env` |
| `src/runtime.py` | Builds the real dependencies (HTTP clients, embedder, brains) from settings |
| `src/api/app.py` | HTTP API: create, list, approve and cancel jobs; serve paper, references, report and trace |
| `src/worker.py` | Worker process: claim jobs, heartbeat, run the graph, survive errors |
| `src/observability.py` | JSON logs, Prometheus metrics, and the per-job context |

**Agent**

| File | What it does |
|---|---|
| `src/agent/graph.py` | Wires the steps into a LangGraph graph |
| `src/agent/nodes.py` | One method per step: load inputs, call the logic, save outputs |
| `src/agent/state.py` | Graph state: ids and small values only; bulk data lives in Postgres |

**Selection and writing logic**

| File | What it does |
|---|---|
| `src/funnel.py` | Builds the pool, scores it, shortlists, screens, and selects |
| `src/writing.py` | Passages, evidence cards, verification, and the final paper |
| `src/citations.py` | Citation markers, quote checks, reference numbering, BibTeX |
| `src/embed.py` | Turns text into unit-length vectors (fastembed, bge-small) |

**Math** (`src/ranking/`, explained in [math.md](math.md))

| File | What it does |
|---|---|
| `bm25.py` | BM25F keyword scoring over title and abstract |
| `dense.py` | Normalizing, cosine similarity, Rocchio feedback |
| `pagerank.py` | Personalized PageRank by power iteration |
| `fusion.py` | Scores to ranks, and Reciprocal Rank Fusion |
| `bayes.py` | Turns the judge's verdict into a probability |
| `select.py` | Top-k, MMR, submodular objectives, lazy and budgeted greedy |
| `dpp.py` | DPP selection by incremental Cholesky |
| `cluster.py` | Spectral clustering and k-means |
| `metrics.py` | Recall, nDCG, MRR, bootstrap intervals, permutation test, kappa |

**Brains**

| File | What it does |
|---|---|
| `src/brain/base.py` | The Brain interface and the data shapes it uses (Plan, Verdict, EvidenceCard...) |
| `src/brain/offline.py` | Free, rule-based stand-ins for every LLM step |
| `src/brain/claude.py` | Each step as one Anthropic Messages API call, with structured output, caching and fallbacks |

**Data in and out**

| File | What it does |
|---|---|
| `src/sources/http.py` | One HTTP client for outside APIs: rate limits, retries, disk cache, cost tracking |
| `src/sources/openalex.py` | OpenAlex: search, fetch by id, citing works, samples |
| `src/sources/arxiv.py` | arXiv: find a paper by title, download its PDF |
| `src/text/pdf.py` | PDF bytes to clean text (the PDF is never stored) |
| `src/text/chunking.py` | Section-aware, sentence-packed chunks |
| `src/text/tokenize.py` | Tokens, token counts, sentence splits, normalization for quote matching |
| `src/db/schema.sql` | Tables and indexes (pgvector HNSW, full-text GIN) |
| `src/db/jobs.py` | The job queue, leases, events, LLM costs |
| `src/db/papers.py` | Papers and chunks, vector and keyword search |
| `src/db/results.py` | Per-job scores, sections, evidence cards, outputs |

**Outside `src/`**

| Path | What it holds |
|---|---|
| `evals/` | `selection_eval.py` (surveys as answer keys), `writing_eval.py` (checks finished jobs), `results/` |
| `lab/` | The vector search lab: `lsh.py`, `ivf.py`, `pq.py`, `hnsw.py`, and `bench.py` to compare them |
| `samples/` | Output of a real run: paper, references, report, selection, trace |
| `docs/` | Architecture, math, study guide, interview prep, and this tour |
| `Dockerfile`, `docker-compose.yml` | The image and the three services (postgres, api, worker) |

## 6. Where the tests are

Each test file sits next to the area it checks, mirroring `src/`:

| Area | Tests |
|---|---|
| Ranking math | `tests/ranking/` (one file per method or pair of methods) |
| Selection and writing logic | `tests/test_funnel.py`, `tests/test_writing.py`, `tests/test_citations.py` |
| Text | `tests/text/test_text.py` |
| Outside APIs | `tests/sources/` (fake servers, no network) |
| Brains | `tests/brain/` (the Claude brain runs against a fake Anthropic API) |
| Database | `tests/db/` (marked `db`; needs Postgres) |
| Full jobs, crashes, cancel, budget | `tests/agent/test_graph.py` (marked `db`) |
| API, worker, config, evals, lab | `tests/api/`, `tests/test_worker.py`, `tests/test_config.py`, `tests/test_evals.py`, `tests/lab/` |
| Shared fakes | `tests/fakes.py` (a fake OpenAlex world, fake arXiv), `tests/conftest.py` |

To run one file or one test:

```bash
uv run pytest tests/ranking/test_pagerank_bayes.py -q
```

```bash
uv run pytest -k "lazy_greedy" -q
```

## 7. Where to make common changes

| I want to... | Go to |
|---|---|
| Change a ranking or writing parameter | `AlgoParams` in `src/config.py` |
| Change what a job request accepts | `JobSettings` in `src/config.py`, then `src/api/app.py` |
| Change a prompt or a model setting | `src/brain/claude.py`; per-step models and effort via `MODEL_*` and `EFFORT_*` settings |
| Add a step to the workflow | a method with `@step` in `src/agent/nodes.py`, then wire it in `src/agent/graph.py` |
| Add a ranking signal | a new file in `src/ranking/`, then `funnel.compute_signals()` |
| Add a paper source | a client in `src/sources/` built on `HttpClient`, then `funnel.gather()` |
| Change the database | `src/db/schema.sql` (applied at startup) and the matching file in `src/db/` |
| Measure whether a change helps | `uv run python -m evals.selection_eval` before and after |

## 8. House rules you'll see in the code

- **Steps are safe to run twice.** A crash can rerun the step that was in progress, so writes are insert-or-update and a paper with chunks is not ingested again.
- **Graph state stays small.** It holds ids, not papers or drafts, so checkpoints stay fast.
- **The model never writes a reference.** It cites evidence card ids; code turns those into numbered references from metadata.
- **Every quote is checked word for word** against its source chunk before anyone can cite it.
- **Tests come first.** Each behavior has a test that failed before the code existed. Math is tested by its properties (PageRank sums to 1, lazy greedy equals plain greedy), not just by examples.
