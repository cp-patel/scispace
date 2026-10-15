# Session context

This file carries the working context of the chat sessions that built this repository, so a new session can pick up where the last one stopped.

**For the assistant in a new session:**
1. Read this whole file before doing anything.
2. Then check the "Open items" in the latest entry against the real state (`git log`, `evals/results/`, `lab/results/`). Things may have finished, or failed, since the entry was written.
3. When asked to update this context, **append** a new dated entry at the bottom under "Context updates". Never edit or delete earlier entries. If something in an earlier entry is now wrong, say so in the new entry.

---

## Baseline (written 2026-10-15)

### Who and why

- **The user:** Chandra. Not deeply technical yet, wants to grow technical skills, and is strong at math.
- **The goal:** prepare for a SciSpace interview about a week after 2026-10-14. The role is titled "Senior Research Scientist", but it is really a GenAI ML-systems engineering role (Python, ML services and APIs, LLM deployment, Docker, data pipelines, SQL, testing, LangGraph, cloud, distributed systems).
- **The interview round's question:** "Given thousands of relevant papers for a topic, how would you evaluate and select the most pertinent ones, and what would the agentic architecture look like to synthesize them into a well-cited research paper?"
- **The project:** a working, production-grade answer to that question, plus study material to defend it.

### How Chandra wants the assistant to work

These come from Chandra's global instructions and from this session.

- **Writing:**
  - Use a sixth-grade reading level: short sentences and everyday words, but never drop a real detail, number, or caveat.
  - Use real terms when needed; define them in the same breath with an analogy the first time.
  - Explain the why, not just the what.
- **Evidence:** when claiming something works, show the evidence (command output), not just the claim.
- **Outward content** (commits, PRs, shipped docs, Slack, tickets):
  - Never use em-dashes.
  - Never add AI attribution or `Co-Authored-By` lines to commits.
  - PR descriptions are clean current-state summaries.
- **Code comments** explain what and why, never where a requirement came from (no ticket ids, no "per the plan").
- **Planning documents** go in `~/Documents/thoughts/`, with a date prefix.
- **Speed:** Chandra asked to "make it fast a bit". Keep narration short and run independent work in parallel.
- **Security:**
  - Never print the `API_KEY` or any other secret value; only check whether it is set.
  - `.env` stays git-ignored.
  - Don't create accounts or enter credentials.

### Decisions Chandra made

- **Scope:** about a week, so the core system plus a production shell (API service, worker, Postgres, Docker Compose, evals, cost tracking).
- **LLM access:** no API key at first. An "offline brain" (rule-based stand-in) runs everything for free; a Claude brain (default model `claude-opus-5`) switches on later with no code change. Tests mock the Claude API.
- **Architecture:** "approach A", Postgres for everything (queue, papers, vectors, checkpoints, results). Two processes: API and worker.
- **Diagram:** in the style of Chandra's own RAG diagram (dark navy, indexing pipeline on top, query side below). Every topic from Chandra's study notes is mapped to the project.
- **Math to show off:** hand-written BM25F, vector search, PageRank and more. Chosen extras: a vector search lab (LSH, IVF, PQ, HNSW), DPP selection, and spectral clustering for the outline. Learned fusion was not chosen.
- **Process:** Chandra approved the design section by section, approved the written spec and then the plan, and chose to "build natively" (the assistant implements every task itself, then one fresh reviewer checks the whole branch).
- **Branch:** after the build, Chandra asked to put everything on `main`.
- **GitHub:** Chandra created https://github.com/cp-patel/scispace (public) and asked for the code to be pushed there.

### Documents outside this repository

- **Spec (the binding design):** `~/Documents/thoughts/2026-10-14-research-assistant-design.md`
- **Implementation plan:** `~/Documents/thoughts/2026-10-15-research-assistant-plan.md` (27 tasks, generated from a validated source tree, so the plan's code equals the tested code).
- **Interview prep page (private Claude artifact):** https://claude.ai/artifact/PkbBREDBKkjtLe1k97XhrT, titled "Research Assistant Playbook". It covers the 60-second answer, architecture, the math, eval results, a traced job, two bug stories, weak spots, a 45-question drill with saved marks, the five-day plan, the job-post mapping and a glossary.
  - **To update it from another session:** read it with the Artifact tool's `read` action on that URL, edit the HTML, and publish with `url` set to the same link.
  - Its question bank, job-post table and glossary come from `docs/study-guide.md`.

### What the system does

- **Selection funnel** (`src/research_assistant/funnel.py`, math in `ranking/`):
  - **plan:** the topic becomes facets (sub-questions).
  - **gather:** the candidate pool comes from the local pgvector store, plus an OpenAlex keyword search per facet, plus citation snowballing.
  - **rank:** three signals are merged with weighted Reciprocal Rank Fusion (`k = 60`; facet lists weigh `1/F` each, PageRank weighs 1). The signals are:
    - **BM25F:** title `w = 2, b = 0.3`; abstract `w = 1, b = 0.75`; `k1 = 1.2`.
    - **Dense cosine with Rocchio:** alpha 1, beta 0.75, top 20.
    - **Personalized PageRank:** alpha 0.85, teleport to the top 50 text scores.
  - **screen:** a 150-paper shortlist goes to the judge, and Bayes' rule turns each verdict into a probability. It uses the judge's sensitivity and specificity, with a calibration table by rank as the prior.
  - **select:** picks K = 40. The default is a submodular objective (facet coverage plus facility location) solved with lazy greedy. MMR, DPP (greedy MAP via incremental Cholesky) and top-k are alternatives.
- **Writing** (`writing.py`, `agent/`):
  - **ingest:** arXiv PDFs are read in memory and never stored (1 request per 3 s). Chunks are section-aware and sentence-packed: 500 tokens, 50 tokens of overlap, back matter cut.
  - **outline:** spectral clustering picks the sections (`W = 0.7 cos + 0.3` bibliographic coupling; eigengap k in [3, 8]).
  - **Per section, in parallel through LangGraph `Send`:** retrieve, then compress into evidence cards (budget 6,000 tokens, verbatim quotes, ids like `E2.3`), then write, then verify, revise up to 2 times, and finish.
  - **assemble:** builds the references from metadata only.
- **Production shell:**
  - **API:** FastAPI with an `X-API-Key` header.
  - **Job queue:** a Postgres table claimed with `FOR UPDATE SKIP LOCKED`. Leases last 120 s, heartbeats come every 30 s, and a reaper allows 3 attempts.
  - **Recovery:** LangGraph checkpoints in Postgres (`AsyncPostgresSaver`, `thread_id = job_id`), plus a human approval pause via `interrupt()`.
  - **Calls out:** retries with jitter, rate limits, a disk HTTP cache, and per-job LLM budgets.
  - **Observability:** JSON logs, Prometheus metrics, and a per-job trace.
  - **Docker:** Compose runs Postgres 18 with pgvector on host port 55432, plus the api and worker.
- **Key versions:** LangGraph 1.2.15, Anthropic SDK 1.14, pgvector 0.8.7, fastembed `BAAI/bge-small-en-v1.5` (384 dimensions).
- **Docs in the repo:**
  - `docs/architecture.md` explains the system.
  - `docs/architecture-rag.svg` is the diagram.
  - `docs/math.md` covers every algorithm with proofs and measured numbers.
  - `docs/study-guide.md` has the five-day plan, a traced run, 45 questions with answers, the job-post mapping, the notes mapping and a glossary.

### How the build went

- All 27 plan tasks were built test-first (each test watched failing, then passing), with a progress ledger.
- A fresh reviewer then checked the whole branch. Verdict: "With fixes". It found 1 Critical and 9 Important issues, all fixed in one pass, each with a new test that failed before the fix (commit `bb3e344`):
  - **Critical:** source papers' own citation numbers (like `[14]`) leaked into our paper and looked like our references. They are now stripped in PDF cleaning and in marker rendering, and the writing eval counts numbers beyond the reference list.
  - **Important:**
    - Screening uses an `asyncio.TaskGroup`, so a budget failure cancels the other batches.
    - A sentence the support judge skips counts as unsupported.
    - Local-store hits obey the year range.
    - The worker heartbeat and main loop survive database errors, and Compose uses `restart: unless-stopped`.
    - `calibration.json` ships in the Docker image, and a missing file logs a warning.
    - An approval that removes every paper fails the job with `NoPapersFound`.
    - A broken arXiv feed counts as "not found".
    - Numbered or lettered References and Appendix headings end the body.
    - Quotes with glued PDF words (30+ letters) are dropped.
- **Deferred minor items (not fixed; Chandra decides):**
  1. `.env.example` caches OpenAlex results forever.
  2. Paper id formats are not validated at the API.
  3. `/metrics` needs no key.
  4. `/paper` returns 404 instead of 409 for an unfinished job.
  5. Unknown models are priced as Opus.
  6. The report's selection table stops at 300 rows.
  7. A pgvector session setting leaks to pooled connections.
  8. The deadline applies per attempt, not per job.
  9. Drafts go into prompts unescaped.
  10. `math.md` says the calibration table is monotone, but its last bucket dips.
  11. The worker metrics port is fixed at 9100.
  12. The report repeats steps after a resume.
  13. The default 500 error body differs from the API's error shape.
  14. PageRank doesn't check edge indices.
  15. Card ids are section-scoped, which the spec doesn't mention.
  16. The SDK's own retries replaced the spec's typed exception chain.
  17. Quotes with PDF math symbols can reach the paper.

### State at the end of this entry

- **Git:** branch `main`, pushed to `origin` (https://github.com/cp-patel/scispace). The latest pushed commit is `a587795`, which merged GitHub's initial README commit and kept the project README. No force push was used.
- **Tests:** 304 pass: 280 fast (`uv run pytest`), 23 database (`uv run pytest -m db`, needs `docker compose up -d postgres`) and 1 slow (`uv run pytest -m slow`). `uv run ruff check src tests lab evals` is clean.
- **Docker:** the stack was rebuilt with the fixes (restart policy and calibration file verified, healthz ok).
- **Sample run** (`samples/`): offline brain, OpenAlex answers from the cache, about 80 s.
  - Funnel: 373 candidates, 150 shortlisted, 40 selected; the judge included 90 of 150.
  - Ingest: 17 PDFs, 23 abstract-only papers. Outline: 3 sections (eigengap).
  - Output: 22 evidence cards, 12 references.
  - Writing eval: 0 unresolved markers, 0 bad reference numbers, every quote verified.
- **Selection eval** (7 surveys; recall@150 / nDCG@100):

  | Method | Recall@150 | nDCG@100 |
  |---|---|---|
  | OpenAlex's order | 0.065 | 0.067 |
  | BM25F | 0.080 | 0.054 |
  | Dense | 0.203 | 0.165 |
  | RRF text | 0.186 | 0.129 |
  | Plus PageRank | 0.242 | 0.180 |
  | Full | 0.245 | 0.184 |

  - Significant gains: RRF over BM25F (+0.074, p 0.016) and adding PageRank (+0.051, p 0.032). Rocchio was not significant (+0.004, p 0.45).
  - Recall@40 by selection method: top-k 0.144, MMR 0.141, submodular 0.132, DPP 0.130. DPP picked the most varied sets.
  - Offline judge: sensitivity 0.69, specificity 0.46, kappa 0.06.
- **OpenAlex budget:** keyless, $0.10 per day. Search costs $1 per 1,000 calls, list calls $0.10 per 1,000, single lookups are free. `HTTP_CACHE_DIR=.http-cache` makes repeat calls free.

### Open items at the end of this entry

1. **8-survey eval and real 50K benchmark.** An overnight timer was meant to start them at 05:33, but the Mac slept and macOS paused the timer. At 10:15 the timer was stopped and both were started directly:
   ```bash
   HTTP_CACHE_DIR=.http-cache uv run python -m evals.selection_eval
   ```
   ```bash
   HTTP_CACHE_DIR=.http-cache uv run python -m lab.bench --source openalex --n 50000 --queries 1000 --pgvector
   ```
   When they finish:
   - Commit `evals/results/` (8 surveys, new `calibration.json`) and `lab/results/`.
   - Update the numbers in `docs/math.md`, `docs/study-guide.md`, `README.md` and the prep page.
   - Rebuild the Docker image so it picks up the new calibration table.
   - Push to GitHub.
2. **This file** is committed locally. Pushing it to the public GitHub repo was left for Chandra to decide.

---

## Context updates

(Append new dated entries below. Do not edit the entries above.)

### 2026-10-15, late morning: prep docs, 8-survey eval, code tour

**Corrections to the baseline entry**
- **The interview prep document is now a repo file:** `docs/interview-prep.md`. Chandra asked for prep material to live in the repository, not as a Claude artifact. The playbook artifact named in the baseline is obsolete.
  - Chandra agreed to delete that artifact, but the delete needs an on-screen confirmation the session couldn't show. Chandra can remove it from claude.ai (the artifact's menu) or with `/artifacts` in a Claude Code terminal (press d).
  - Rule for the future: project documents go in `docs/` as Markdown, linked from the README, then committed and pushed.
- **The 8-survey eval ran.** The baseline's eval numbers (7 surveys) are superseded by the table below.

**What changed**
- `SESSION_CONTEXT.md` was pushed to GitHub, at Chandra's request.
- `docs/interview-prep.md` was added:
  - the 60-second answer, the architecture with a Mermaid diagram, and the math per method;
  - eval tables, a traced job, two bug stories, weak spots, and the five-day plan;
  - a 45-question drill with collapsible answers, generated from section 4 of `docs/study-guide.md`. Keep the two in sync: if a study-guide answer changes, change the drill too.
- `docs/code-tour.md` was added: how to run it, the pure-logic vs I/O layout, a 10-stop reading order, one job traced through the code with function names, a file map, a test map, and where to make common changes. Linked from the README.
- **The overnight timer failed.** The Mac slept, and macOS pauses `sleep` timers during sleep. The timer was stopped and both runs were started directly.
  - **8-survey eval:** succeeded, at an OpenAlex cost of $0.016. `evals/results/` and `calibration.json` were committed, and the numbers were updated in `docs/math.md`, `docs/study-guide.md` and `docs/interview-prep.md`.
  - **Real 50K benchmark:** failed with `401 API key required for cursor paging`. OpenAlex now requires a free API key to page through large result sets. Chandra has to create it at openalex.org (an account sign-up the assistant must not do) and add `OPENALEX_API_KEY=...` to `.env`. Then run:
    ```bash
    HTTP_CACHE_DIR=.http-cache uv run python -m lab.bench --source openalex --n 50000 --queries 1000 --pgvector
    ```
    Until then, the docs say the benchmark numbers are synthetic.
- The Docker image was rebuilt with the 8-survey calibration table (verified inside the worker container).

**Current numbers** (8 surveys from 7 fields; values with intervals are means with 95% bootstrap intervals)

| Ranking | Recall@150 [95% interval] | nDCG@100 |
|---|---|---|
| OpenAlex's order | 0.059 [0.030, 0.090] | 0.064 |
| BM25F | 0.081 [0.042, 0.125] | 0.067 |
| Dense | 0.205 [0.181, 0.232] | 0.184 |
| RRF text | 0.188 [0.170, 0.212] | 0.150 |
| Plus PageRank | 0.241 [0.200, 0.296] | 0.200 |
| Full | 0.245 [0.198, 0.311] | 0.204 |

- **Significant (nDCG@100):** RRF over BM25F, +0.083 (p 0.008); adding PageRank, +0.050 (p 0.014).
- **Not significant:** adding Rocchio, +0.004 (p 0.40).
- **Recall@40 by selection method:** top-k 0.164, MMR 0.154, submodular 0.150, DPP 0.151. None of the differences are significant. DPP picks the most varied sets (mean pairwise cosine 0.777, vs 0.828 for top-k).
- **Offline judge:** sensitivity 0.71, specificity 0.46, kappa 0.08.
- **Calibration prior** by shortlist rank:

  | Ranks | Prior |
  |---|---|
  | 1-10 | 0.15 |
  | 11-25 | 0.13 |
  | 26-50 | 0.18 |
  | 51-100 | 0.19 |
  | 101+ | 0.08 |

- **Tests:** 304 pass (280 fast, 23 database, 1 slow). Lint is clean.

**State at the end of this entry**
- Branch `main` is in sync with `origin`: https://github.com/cp-patel/scispace.

**Open items**
1. Run the real 50K benchmark once Chandra has an OpenAlex API key. Then commit `lab/results/`, update the benchmark numbers in `docs/math.md`, `docs/study-guide.md` and `docs/interview-prep.md`, and push.
2. The 17 deferred minor review findings listed in the baseline are still open; Chandra decides which, if any, to fix.

### 2026-10-15, evening: clearer production-shell section

**What changed**
- Chandra found section 5 of `docs/interview-prep.md` ("The production shell") hard to follow; it had been a dense table. It was rewritten in plain terms (commit `e442965`):
  - It opens with why a shell is needed: jobs are slow, depend on outside services, cost money, and can crash.
  - One analogy runs through it: a restaurant kitchen (ticket rail = queue, a cook holding a ticket = lease, notes on the ticket = checkpoint, calling the supplier again = retry, spending limit = budget).
  - It is organized around four problems:
    1. Slow work: the job queue and `SKIP LOCKED`.
    2. Crashes: the lease, heartbeat, reaper, checkpoints, and idempotent steps, plus the incident walk-through.
    3. Unreliable outside services: retries with backoff and jitter, rate limits, and the disk cache.
    4. Cost and visibility: the budget, logs, metrics, trace, and cancel and pause.
  - It ends with a Docker part, a 4-sentence "how to say it in the interview" version, and the old table kept as a quick reference.
  - Every claim was checked against the code: the claim query in `db/jobs.py`, the 120 s lease and 30 s heartbeat, 3 attempts, and the retry rule in `sources/http.py`.
- **Writing preference reconfirmed:** Chandra wants explanations that start with the why, use one analogy, and define terms like 429, jitter, and idempotent in plain words. Apply this to future doc sections.

**State at the end of this entry**
- Branch `main` is in sync with `origin`: https://github.com/cp-patel/scispace.
- Tests: 304 pass. No code changed in this entry, only docs.

**Open items (unchanged)**
1. Run the real 50K benchmark once Chandra has a free OpenAlex API key (`OPENALEX_API_KEY` in `.env`).
2. The old playbook artifact still needs Chandra to delete it (claude.ai menu, or `/artifacts` then d).
3. The 17 deferred minor review findings remain open, for Chandra to decide.
