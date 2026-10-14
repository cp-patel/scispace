# Study guide

This guide prepares you to answer one interview question in depth:

> Given thousands of relevant papers for a topic, how would you evaluate and select the most pertinent ones, and what would the agentic architecture look like to synthesize them into a well-cited research paper?

Everything here points at real code, real tests, and real numbers from this repository. Read `docs/architecture.md` for the system and `docs/math.md` for the algorithms.

## 1. The one-minute answer

1. **A funnel.** Cheap methods look at thousands of papers and expensive ones look at few. Keyword search, vectors and the citation graph score about 2,000 candidates in milliseconds; an LLM judge sees only the top 150; full-text reading and writing see only the final 40.
2. **Three kinds of evidence, merged by rank.** BM25F catches exact terms, embeddings catch meaning, and Personalized PageRank catches what the topic builds on. Reciprocal Rank Fusion merges them without calibrating their scales. In our eval, adding the citation graph raised nDCG@100 by 0.051 (p = 0.032).
3. **Selection is a set problem.** Choose 40 papers that are relevant, cover every sub-topic, and are not near-copies: a submodular objective with a proven 63% guarantee, or a DPP that maximizes the volume the set spans.
4. **Evidence cards, not free text.** For each section, retrieve the best passages, pack them into a token budget, and extract claims with quotes. Code rejects any quote that is not in the source word for word.
5. **A verify-and-revise loop.** The writer cites card ids; a checker labels each sentence supported, partial or unsupported; unsupported sentences are rewritten or deleted. References are built from metadata, so a made-up reference cannot happen.
6. **A production shell.** An async API, a Postgres job queue with leases, LangGraph checkpoints so a crashed job resumes on another worker, retries, rate limits, budgets, metrics, and an eval that uses published surveys as answer keys.

## 2. A five-day plan

| Day | Topics | Core or bonus | Read | Run |
|---|---|---|---|---|
| 1 | The funnel story; BM25F; cosine and embeddings; RRF | Core | `math.md` 1, 2, 4; `funnel.py` | `uv run pytest tests/ranking -q` |
| 2 | Personalized PageRank; the Bayes judge; submodular selection and its proof; DPP | Core (DPP bonus) | `math.md` 3, 5, 6, 8; `ranking/select.py` | `uv run pytest tests/ranking/test_select_dpp.py -q` |
| 3 | The agent graph; evidence cards; verification; the job queue, leases and checkpoints; observability | Core | `architecture.md` 4 to 7; `agent/nodes.py`; `worker.py` | `docker compose up -d --build`, then start a job (README) |
| 4 | Evals and what the numbers mean; the vector lab (LSH, IVF, PQ, HNSW) | Core (lab bonus) | `math.md` 10 to 14; `evals/results/`; `lab/` | `uv run python -m lab.bench --source synthetic --n 3000 --queries 100` |
| 5 | Say the answers out loud, timed; draw the diagram from memory | Core | This guide, section 4 | Explain `samples/report.md` in 3 minutes |

## 3. One job, traced end to end

The sample run in `samples/` used the topic "retrieval-augmented generation for scientific question answering", the offline brain, and live OpenAlex and arXiv. It finished in about 3 minutes.

1. **plan** (0.001 s): the topic became 5 facets: the topic itself, plus methods, evaluation, applications, and limitations.
2. **gather** (38.5 s): keyword searches on OpenAlex and citation snowballing found 373 candidate papers. Almost all of this time is waiting on the network.
3. **rank** (0.12 s): BM25F, cosine with Rocchio, and PageRank scored all 373; RRF merged them; the top 150 became the shortlist.
4. **screen** (0.02 s): the judge said "include" for 90 of 150; Bayes' rule turned each verdict into a probability.
5. **select** (0.02 s): submodular lazy greedy picked 40 (objective value 3.39). The top picks include "Re-ranking and Late Interaction Drive Retrieval Quality: A Controlled Comparison of RAG Strategies for Scientific Question Answering".
6. **approve**: skipped, because `auto_approve` defaults to true.
7. **ingest** (123.5 s): 17 papers had arXiv PDFs; 23 fell back to their abstracts. The PDFs were read in memory and never stored. This step is slow on purpose: arXiv asks for one request every 3 seconds.
8. **outline** (0.02 s): spectral clustering of the 40 papers; the eigengap chose 3 sections.
9. **sections, in parallel**: each retrieved its top 40 of 479 chunks, packed about 25 into a 6,000-token budget, kept 8 evidence cards (every quote verified), wrote 190 to 260 words, and passed verification with 0 flags.
10. **assemble** (0.01 s): the abstract, introduction and conclusion were checked too; 11 papers were cited, numbered by first use; BibTeX was built from metadata.

The writing eval on this job: 0 unresolved markers, 24 cards, every quote verified, every sentence cited.

What the offline brain cannot do, and Claude would fix:
- **Section names.** They are word lists, close to each other ("Retrieval, generation, rag").
- **Wording.** The text stitches quotes together ("Smith et al. 2026 report that ...").
- **Coverage.** Only 11 of the 40 selected papers are cited, because it keeps the 8 best-matching sentences per section.

## 4. Question bank

### Selecting papers

1. **How would you pick 40 papers out of thousands?** With a funnel. First get recall: keyword search per sub-question, plus citation snowballing. Then score cheaply with keywords, vectors and the citation graph, and merge the rankings. Then spend an LLM judge only on the top 150. Finally choose the set, not just the top scores, so the 40 cover every sub-question without repeats.

2. **Why a funnel?** Cost and latency grow from step to step: BM25 and vectors cost microseconds per paper, an LLM judgment costs real money and seconds, and reading full text costs minutes. A funnel spends each method where it pays off. In the sample run, the math on 373 papers took 0.15 seconds, while the network took 160.

3. **BM25 or embeddings?** Both. BM25 nails exact rare terms (a gene name, "LoRA"); embeddings catch the same idea in different words. In our eval, vectors found 2.5 times more of the survey-cited papers than keywords did, but keywords stay essential for exact-term queries. Hybrid search is the standard answer.

4. **When does keyword search win?** Rare, exact, high-information tokens: identifiers, chemical names, acronyms, version numbers. Embeddings smooth these into nearby concepts, so a query for one specific model name can return its cousins.

5. **Why Reciprocal Rank Fusion?** The scores live on different scales, so averaging them is meaningless without calibration. RRF uses only ranks, `1 / (60 + rank)`, so it is scale-free and robust. The trade-off: it ignores how big a lead is, and equal weights can let a weak list pull down a strong one, as our eval showed. Learned weights fix that.

6. **What does PageRank add over citation counts?** Citation counts are global popularity, from any field. Personalized PageRank starts its random walks at the strongest papers for this topic, so importance becomes topic-specific, and it finds foundational papers that use different words. It gave the largest significant gain in our eval.

7. **How do you combine an LLM judge with retrieval scores?** Treat the judge as a noisy test with measured sensitivity and specificity. Start from a prior based on the paper's rank, and update it with Bayes' rule. A weak judge barely moves the prior, and a strong one moves it a lot. Our offline judge had kappa 0.06, and the math correctly mostly ignored it.

8. **Why is selection a set problem?** A paper's value depends on what else is chosen: the second paper on the same method adds less than the first. Top-k by score gives near-duplicates. A submodular objective rewards covering every facet and every part of the candidate space.

9. **Prove the greedy bound.** For a monotone submodular F, the gap to the optimum after step i satisfies `gap(i+1) <= (1 - 1/k) gap(i)`, because the optimal set's items together could add at least the gap, and greedy took the largest single gain. After k steps the gap is at most `(1 - 1/k)^k <= 1/e` of the optimum, so greedy reaches at least 63%.

10. **MMR, submodular or DPP?** MMR is a quick rule with no guarantee. Submodular lets you write down what "good set" means and gives a guarantee. DPP scores a whole set as volume, quality times diversity, but its greedy has no clean guarantee. In our eval DPP produced the most varied sets (mean pairwise cosine 0.78 vs 0.82), while recall differences were not significant.

11. **How do you evaluate selection without labels?** Use published surveys as answer keys. A survey's reference list is an expert's selection for a topic. Run the pipeline as if on the survey's publication date, exclude the survey itself, and measure recall of its references. We did this for 7 surveys in 6 fields, with bootstrap intervals and permutation tests.

12. **What are the limits of a survey as an answer key?** A survey does not cite every good paper, so precision is only a floor and recall is the trustworthy number. A survey's taste may differ from what a user wants. It rewards finding cited papers, not variety, so it cannot measure diversity. And with a handful of surveys, the intervals are wide.

13. **How confident are you in your numbers?**
    - **What is solid:** with 7 surveys, two improvements are significant (fusion over keywords alone, and adding PageRank), with p-values of 0.016 and 0.032.
    - **What is not:** the Rocchio and selection-method differences are within noise.
    - **The caveat I'd state:** absolute recall is low (about 0.25 at 150), mainly because the offline planner's keyword queries are narrow. An LLM planner is the first thing I'd measure.

### Reading and writing

14. **How do you chunk papers, and why?** Split at section headings, then pack whole sentences into windows of about 500 tokens with about 50 tokens of overlap. Sections keep each chunk about one thing. Whole sentences mean a quote is never cut in half. The overlap means a claim near a boundary appears whole somewhere. Alternatives: fixed-size windows (simple, but they cut meaning) and semantic chunking (split where embedding similarity drops).

15. **How do you fit evidence into the context window?** Each section gets its own small evidence pack: retrieve the top 40 passages, then pick a subset under a 6,000-token budget with budgeted submodular selection, then compress it into evidence cards. Sending all 40 full papers would cost more, take longer, and models miss details buried in the middle of long contexts.

16. **What is contextual compression here?** Turning retrieved passages into evidence cards: one claim plus the exact supporting quote. The writer sees only those cards, which saves tokens and makes every claim checkable.

17. **How do you prevent made-up references?** The LLM never writes a reference. It cites card ids like `[E2.3]`; code maps each id to a paper and builds the reference from OpenAlex metadata. An id that does not exist is flagged, and its sentence is deleted.

18. **How do you check that claims are supported?**
    - **Code checks first:** every cited id exists in this section, every paragraph has a citation, and long sentences without a citation are flagged.
    - **Then the brain:** it labels each cited sentence supported, partial, or unsupported, judging only against its quotes.
    - **Then repair:** flagged sentences get up to 2 revision rounds, and whatever still fails is deleted.
    - **Why it holds up:** every quote was already verified word for word against its chunk.

19. **What happens when the LLM refuses or returns broken output?**
    - **Refusals:** server-side fallback reroutes them to another model. If the whole chain refuses, the step raises an error.
    - **Cut-off answers:** an answer that hits max tokens raises too.
    - **Broken content:** malformed citations like `[E3]`, quotes not in the source, and duplicate quotes are caught by code and removed. The section is dropped only if fewer than 2 supported sentences remain, and the job still finishes.

20. **What about prompt injection hidden in an abstract?**
    - **Containment:** paper text is escaped and wrapped in tags, and the system prompt says tagged text is data.
    - **Structure:** outputs are structured, so injected text cannot change the shape of the answer.
    - **Verification:** quotes must match the source, and references come only from metadata.
    - **Result:** the worst an injected abstract can do is get itself screened in, and the citation checks still apply.

### The agent and LangGraph

21. **Why LangGraph instead of a plain loop?** The workflow is a known graph with a loop (verify and revise) and a fan-out (sections in parallel). LangGraph gives you that graph explicitly, plus checkpoints after every step, a human pause with `interrupt()`, and parallel branches with `Send`. A free-roaming agent would be harder to test, bound, and recover.

22. **LangChain or LangGraph?** LangChain offers building blocks (model wrappers, prompts, retrievers); LangGraph runs stateful graphs of steps. This project uses LangGraph for orchestration and calls the Anthropic SDK directly for model calls. That keeps full control of structured outputs, caching and fallbacks, with fewer layers.

23. **How does crash recovery work?** LangGraph saves a checkpoint to Postgres after every node. If a worker dies, its lease expires, the reaper requeues the job, and another worker invokes the graph with the same thread id, so it continues from the last finished node. The crash test kills a worker during `outline` and checks that `gather` and `ingest` did not run again.

24. **How do you keep a human in the loop?** The `approve` node pauses the graph after selection, before the expensive full-text and writing steps. A person can remove or add papers through the API, and the worker resumes with their answer.

25. **Why keep graph state small?** Checkpoints are written after every step. State holds ids and small values; papers, chunks and drafts live in tables. Checkpoints stay fast, and big data is stored once.

### Production

26. **How do you make sure a job runs exactly once?** Workers claim jobs with `SELECT ... FOR UPDATE SKIP LOCKED`, so two workers can never grab the same row. A lease plus a heartbeat proves the worker is alive. Every step checks it still holds the lease before running, and every write is idempotent, so a rare overlap cannot corrupt data.

27. **Postgres as a queue: where does it stop working?** At very high job rates, or when many consumers need the same event. Then the polling and row locks become the bottleneck, and SQS or Kafka fits better. Research jobs are slow and few, so Postgres is the simpler right choice here, and it gives transactions across queue and results for free.

28. **What does idempotent mean, and where does it matter?** An operation is idempotent when running it twice has the same effect as running it once. After a crash, the interrupted step runs again, so every write is an insert-or-update keyed by job and item, and chunk replacement is one transaction.

29. **How do retries work?** Retry 429 and 5xx responses and connection errors, with exponential backoff and full jitter: wait a random time between 0 and min(30 s, 2^attempt s). Honor `Retry-After`, and stop after 5 tries. The randomness keeps many clients from retrying in lockstep. Do not retry 4xx errors other than 429: they will not fix themselves.

30. **How do you respect outside rate limits?** A shared rate limiter per service spaces requests: OpenAlex at 8 per second (its limit is 10), and arXiv at one request every 3 seconds on one connection (its published rule). A disk cache makes repeated calls free, which matters because OpenAlex now charges per call beyond a daily allowance.

31. **How do you control LLM cost?**
    - **Measure:** every call records its tokens and dollar cost against the job.
    - **Cap:** a job stops once it passes its budget (default $10).
    - **Fewer tokens:** prompt caching reuses long fixed instructions across screening batches, and each section's evidence is capped at 6,000 tokens.
    - **Spend effort only where it pays:** effort is set per step (low for screening, high for writing).
    - **Cheaper options to consider:** the Batch API halves the price for non-urgent screening, and a cheaper model can take a step once evals show quality holds.

32. **Which model where?** One strong model (Claude Opus 5) for everything by default, with effort tuned per step. A cheaper model for bulk screening is an option to measure, not assume: the eval harness can compare brains on the same surveys.

33. **What do you monitor?**
    - **Logs:** JSON lines tagged with job and step.
    - **Metrics:** jobs by status, time per step, LLM tokens and cost, outside-API errors, and queue depth.
    - **Trace:** a per-job timeline through the API.
    - **What I'd alert on:** queue depth rising, failure rate, cost per job, and the share of sections dropped.

34. **How would you test a system like this?**
    - **Math as properties:** PageRank sums to 1, greedy meets its bound, and lazy greedy matches plain greedy.
    - **Pure logic against a fake world:** fake OpenAlex and arXiv servers, so runs are free and repeatable.
    - **The Claude brain against a fake API:** this checks the wiring, not quality.
    - **End to end on real Postgres:** full jobs, including a worker crash, cancel, and the budget cap.
    - **Evals for quality, kept separate from tests:** a test checks correctness, an eval measures quality.

35. **Walk me through an incident: a worker dies in the middle of a job.**
    1. The job stays `running` with a lease that is no longer renewed.
    2. Within about 2 minutes the lease expires, and any worker's reaper moves the job back to `queued`, counting the attempt.
    3. Another worker claims it, sees checkpoints exist, and resumes from the last finished step.
    4. If the same job keeps killing workers, it fails after 3 attempts with a clear error, instead of looping forever.

### Vector search

36. **HNSW, IVF, PQ or LSH?**
    - **HNSW:** best recall per query time, but it stores full vectors plus links, so it uses a lot of memory.
    - **IVF:** simple and tunable with `nprobe`.
    - **PQ:** compresses vectors 32 times; you re-rank to recover recall.
    - **LSH:** has clean theory, but needs many tables for high recall.
    - **At scale:** IVF-PQ or a disk-based graph, with a full-vector re-rank.

37. **What did your benchmark show?**
    - **On 3,000 synthetic vectors, exact search was fastest** (about 32,000 queries per second), because indexes only pay off at larger sizes.
    - **HNSW and IVF with enough probes reached recall 1.0.**
    - **PQ alone reached 0.40 at 48 bytes per vector**, and 1.0 after re-ranking 100 candidates.
    - **LSH needed 16 tables to pass 0.9.**
    - **Still to come:** the run on 50,000 real abstracts is the next step (`lab/bench.py --source openalex`).

38. **Pre-filter or post-filter in vector search?** Post-filtering (find nearest, then drop rows that fail a filter like a year range) can return too few results. Pre-filtering (filter first, then search) cannot use the ANN index well. pgvector's iterative scans search the index and keep going until enough rows pass, which this project turns on.

39. **What is data compression in this system?**
    - **Contextual compression:** evidence cards keep only the quotes that matter.
    - **Vector compression:** PQ, or `halfvec` half-precision floats, shrinks vectors for large indexes.

### Scale and next steps

40. **How would this scale to 200 million papers?**
    - **Embed ahead of time:** embed the whole corpus in batch jobs, plus daily updates for new papers.
    - **Shard the indexes:** sharded, compressed ANN indexes, with an exact re-rank; BM25 in a search engine.
    - **Precompute the citation graph** with global PageRank.
    - **Cache screening verdicts** per paper and criteria.
    - **Self-host models** for embeddings and screening.
    - **Autoscale workers on queue depth**, and add a dedicated queue once Postgres becomes the bottleneck.
    - The rest of the design (funnel, graph, verification) stays the same.

41. **Why self-host open-source models, and how?**
    - **Why:** for high-volume, simple steps like embeddings, reranking or screening, self-hosting cuts per-call cost and keeps data in-house.
    - **How:** package the model in a Docker image (this repo bakes its embedding model in at build time), serve it with something like vLLM on GPU nodes, and autoscale.
    - **Keep the hosted model for writing**, where quality matters most.

42. **What is MCP, and how could this system use it?** The Model Context Protocol is a standard way for AI assistants to call tools. This system could expose `search_papers`, `start_review` and `get_review` as MCP tools, so a chat assistant could run a literature review on request. It would be a thin layer over the existing API.

43. **What would you build next?**
    - **An LLM planner:** the first fix for recall.
    - **Learned fusion weights:** logistic regression on the survey labels.
    - **A hand-labeled set** to check the LLM judge.
    - **Better full text:** GROBID for paper structure.
    - **Coverage labels** to evaluate diversity.
    - **An MCP server** over the API.

44. **How would Claude change the results?**
    - **Better facets and keywords:** a bigger, better pool, so higher recall.
    - **A real judge:** much higher kappa than the offline rules' 0.06.
    - **Real writing:** sections named by meaning, real synthesis instead of stitched quotes, and more papers cited.
    - **Nothing to rebuild:** the evals and verification stay the same, so the comparison is fair.

45. **What would you do differently with more time?**
    - **Measure the planner first:** recall is limited by the pool.
    - **Collect a small set of human labels**, for both relevance and claim support.
    - **Test resilience in staging:** run the system under failure injection, such as killing workers and throttling APIs.

## 5. The job post, line by line

| Job post line | Where this repository shows it |
|---|---|
| ML system development: services and APIs | `api/app.py`, `worker.py`, `funnel.py` |
| Model deployment, including transformer LLMs | `embed.py` (a transformer embedding model baked into the Docker image), `brain/claude.py` (fallbacks, retries, budgets) |
| Infrastructure management | `Dockerfile`, `docker-compose.yml`, Postgres with pgvector, worker scaling |
| Data pipelines | `sources/`, `text/chunking.py`, the `gather` and `ingest` steps, `lab/data.py` |
| Collaboration with researchers | `evals/` (shared answer keys), the per-job report, `docs/` |
| Performance optimization | lazy greedy, caching, rate limits, the vector lab benchmark |
| Documentation | `docs/`, `README.md`, interactive API docs at `/docs` |
| Designing multi-component systems | API, worker, queue, database, agent: `docs/architecture.md` |
| Python | the whole repository, typed, lint-clean |
| SQL and database design | `db/schema.sql`, `db/jobs.py` (`SKIP LOCKED`), indexes and iterative scans |
| Testing fundamentals | 291 tests: math properties, fakes, end-to-end, crash and race tests |
| Communication | this guide; `docs/math.md` explains each algorithm plainly |
| Managing technology products | spec, plan, evals, and honest results |
| Gen AI approaches | RAG, LLM-as-judge with Bayes, structured outputs, citation verification |
| Agentic architectures with LangGraph | `agent/graph.py`: fan-out, loops, interrupts, checkpoints |
| Bonus: always-available web apps | health and readiness checks, metrics, async jobs, graceful shutdown |
| Bonus: cloud | `docs/architecture.md` section 10 |
| Bonus: open-source models in Docker | the embedding model baked into the image; the vLLM plan for scale |
| Bonus: distributed systems | leases, heartbeats, idempotent steps, exactly-once claims, crash resume |

## 6. Your notes, mapped

| Topic | Where it lives |
|---|---|
| Overall RAG architecture | `docs/architecture-rag.svg`, `docs/architecture.md` section 2 |
| Chunking | `text/chunking.py` (section-aware, sentence-packed, overlapping) |
| Indexing | HNSW and GIN indexes in `db/schema.sql`; our own HNSW in `lab/hnsw.py` |
| Data compression | evidence cards (`writing.py`); PQ in `lab/pq.py` |
| LangChain vs LangGraph | question 22; `agent/graph.py` |
| MCP | question 42 |
| Vector vs vectorless search | `funnel.compute_signals` (BM25F plus cosine), `PgLocalSearch` |
| Vector search | cosine in `ranking/dense.py`; pgvector; `lab/` |
| Vectorless search | BM25F in `ranking/bm25.py`; citation snowballing in `funnel.gather` |
| Filtering | year and date filters in `sources/openalex.py`; iterative scans in `db/papers.py` |
| Query side | the bottom half of the diagram; `funnel.py` |
| Prompt engineering | the prompts in `brain/claude.py` |
| Query transformation | facets in the `plan` step; Rocchio in `ranking/dense.py` |
| Retrieval algorithms | BM25F, dense, PageRank, LSH, IVF, PQ, HNSW |
| Ranking fusion | `ranking/fusion.py` (RRF) |
| AI hallucination | `citations.py`, `writing.verify`, `writing.finalize` |
| Observability | `observability.py`, `/metrics`, `/v1/jobs/{id}/trace` |
| Eval | `evals/selection_eval.py`, `evals/writing_eval.py`, `evals/results/` |
| Improving GenAI classification | the judge as a classifier: sensitivity, specificity, kappa, Bayes (`ranking/bayes.py`) |
| Context window | per-section evidence budgets (`writing.pick_within_budget`) |

## 7. Glossary

- **Ablation**: turning one part off at a time to measure what it adds.
- **ANN (approximate nearest neighbor)**: finding vectors close to a query without comparing against all of them.
- **BM25 / BM25F**: keyword scoring that rewards rare words, stops rewarding repeats, and adjusts for length; the F version weights fields like title and abstract.
- **Bootstrap**: estimating uncertainty by resampling your data many times.
- **Checkpoint**: a saved copy of the agent's progress, like a save point in a game.
- **Cohen's kappa**: agreement beyond chance; 0 is chance, 1 is perfect.
- **Cosine similarity**: how closely two vectors point the same way.
- **DPP (determinantal point process)**: a way to score a set by the volume its vectors span, favoring good and different items.
- **Eigengap**: the biggest jump between consecutive small eigenvalues of a graph matrix; it suggests how many groups the graph has.
- **Embedding**: a list of numbers representing a text's meaning.
- **Evidence card**: a claim plus an exact quote from a source chunk, with an id the writer cites.
- **Facet**: one sub-question of the topic.
- **HNSW**: a layered graph of vectors that search walks toward the query.
- **Idempotent**: safe to run twice with the same effect as once.
- **Lease**: a claim on a job that expires unless renewed.
- **MMR**: a rule that trades relevance against similarity to what is already chosen.
- **nDCG**: a ranking score that rewards relevant items near the top.
- **Personalized PageRank**: importance from random walks that restart at chosen seed papers.
- **PQ (product quantization)**: compressing vectors by replacing pieces with ids of typical values.
- **Recall@k**: the share of relevant items found in the top k.
- **Rocchio feedback**: moving a query vector toward its good results.
- **RRF (reciprocal rank fusion)**: merging rankings by adding `1 / (k + rank)` from each.
- **Submodular**: diminishing returns; adding an item helps less when similar items are already present.
- **Survey as answer key**: using a published survey's references as the papers a good system should find.
