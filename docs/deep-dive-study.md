# Deep Dive Study: PDF Chunking and Vector Ingestion

Interview notes based on this repository's PDF ingestion and chunking path. The implementation references below describe the current code; recommendations are labeled as improvements rather than existing behavior.

## Code path

`src/research_assistant/agent/nodes.py` ingests papers. It extracts PDF text with `extract_text`, cleans it with `clean_text`, calls `chunk_paper`, embeds the chunk texts, and persists chunks and vectors through `replace_chunks`.

- [`pdf.py`](../../src/research_assistant/text/pdf.py): PDF bytes to extracted and cleaned plain text.
- [`chunking.py`](../../src/research_assistant/text/chunking.py): abstract-first, section-aware, sentence-packed chunks.
- [`tokenize.py`](../../src/research_assistant/text/tokenize.py): approximate token counting and sentence splitting.
- [`papers.py`](../../src/research_assistant/db/papers.py): chunk persistence and retrieval.
- [`schema.sql`](../../src/research_assistant/db/schema.sql): chunk/vector and text-search schema.

## 1. Explain the PDF-to-chunk pipeline

### PDF extraction

`extract_text(pdf_bytes)` wraps bytes in `BytesIO`, opens them with `pypdf.PdfReader`, extracts text page by page, substitutes an empty string for pages with no extractable text, and joins pages with newlines. Read errors become `PdfError`.

This is text extraction, not layout reconstruction. The current path does not explicitly represent page coordinates, columns, tables, equations, or reading order. Those details depend on the extractor's output and need validation on representative PDFs.

### Cleanup

`clean_text()`:

1. Expands common ligatures (`ﬁ` → `fi`, etc.) and removes soft hyphens.
2. Joins a word split by a hyphen at a line break (`retriev-\nal` → `retrieval`).
3. Removes numeric citations through `strip_source_citations`.
4. Collapses repeated spaces/tabs and excessive blank lines.
5. Trims outer whitespace while retaining line breaks, which are used by heading detection.

It does not repair arbitrary column-order errors, rebuild tables, or validate that extraction is complete.

### Section detection

`heading()` recognizes numbered lines matching `_NUMBERED` and a fixed set of named headings in `_NAMED`, plus `Abstract`. `split_sections()` walks line by line, starts a new section at each recognized heading, and places later lines in that section. It omits the PDF's Abstract section because the metadata abstract is used as chunk 0. `strip_back_matter()` removes content starting at a recognized References/Bibliography/Acknowledgments/Appendix line.

If no sections are detected, `chunk_paper()` uses the remaining text as a single `Body` section. If at least one section is detected, text before the first recognized heading is currently not collected. Heading recognition is heuristic and may miss headings or mistake body/table lines for headings.

### Sentence-packed windows

`window()` normalizes whitespace, splits into sentences, and packs whole sentences until adding another would exceed `size_tokens` (default 500). When a window closes, it carries whole trailing sentences from that window into the next until adding another would exceed `overlap_tokens` (default 50). Thus overlap is approximate and sentence-granular. Chunk metadata includes ordinal, section, text, and an approximate token count.

An individual sentence longer than the size target is kept whole because the size check only flushes a non-empty current window. Therefore 500 is a target, not a strict maximum. `approx_tokens()` is also an estimate; validate against the embedding model tokenizer and hard input limit before embedding.

## 2. Interview answer: dense PDFs with columns, tables, and equations

> I’d treat extraction and chunking as structure preservation. First I’d inspect reading order and extraction quality on representative pages, especially two-column pages, tables, and equations. The current `pypdf` path produces plain text and does not explicitly preserve layout geometry, so if ordering or table relationships are important I’d add layout-aware extraction or a table-specific representation.
>
> For prose, I’d keep chunks within section boundaries, pack complete sentences to a target token budget, and use modest sentence-level overlap to retain context around chunk boundaries. I’d preserve paper ID, section, ordinal, page/location, and source information as metadata. For tables, I’d keep caption, headers, units, and row labels with the associated values; large tables can be split into row groups with the headers repeated. Equations should retain nearby definitions and explanatory text.
>
> I’d evaluate with retrieval cases that require table facts, boundary-spanning evidence, and passages from different columns. I’d report what the current code guarantees separately from proposed extraction improvements.

## 3. Heading detection: false positives and missed sections

### How to assess risk

Build a manually labeled sample of extracted papers spanning heading styles, disciplines, tables, and PDF layouts. Label candidate lines as headings or non-headings, then measure precision and recall. Inspect errors by type and downstream impact: missed sections may merge unrelated topics; false headings may fragment content or cause text loss. Include examples such as `3.2.1 Ablation Study`, unnumbered `Implementation Details`, table rows, and body lines that resemble numbered headings.

### Improvements to discuss

- Use typography, whitespace, indentation, and coordinates when available; plain text alone has limited evidence.
- Use numbering patterns and sequence consistency as supporting evidence, not absolute rules.
- Consider line length, punctuation, capitalization, and neighboring paragraph/table structure.
- Detect tabular runs (repeated row shapes, aligned numeric cells, separators) to reduce table-row false positives.
- Preserve uncertain lines in the current section rather than making a destructive split.
- Keep pre-heading text in a preamble/body section instead of dropping it when later headings are found.
- Log the evidence/rule for each detected heading and review sampled decisions.

When a table has already been flattened, combine weak signals: repeated line shapes, multiple numeric cells, separators or aligned columns, neighboring rows with the same pattern, and whether the candidate fits a plausible heading sequence. If confidence remains low, avoid splitting and preserve the text.

## 4. Oversized sentences and hard embedding limits

In `window()`, a sentence longer than `size_tokens` is not split when the current window is empty. It is appended whole, so a chunk can exceed the target. This preserves sentence integrity but does not guarantee the embedder's hard limit.

Recommended handling:

1. Count with the embedding model's tokenizer and leave a safety margin.
2. Keep ordinary sentences intact; split an oversized sentence at clauses/semicolons, with smaller boundaries as fallback.
3. Validate every final chunk against the hard limit before embedding.
4. Preserve section/order/source location for split pieces and evaluate whether the split harms meaning.

## 5. Section context and metadata

`Chunk.section` stores the section name, but `window()` gets only the section body. The heading is therefore not part of `Chunk.text`. Make section metadata available to retrieval and generation. If experiments show heading text improves semantic recall, test a compact embedding prefix against metadata-only retrieval; avoid repeatedly embedding long hierarchy strings without evidence of benefit.

## 6. Data sanity checkpoints before vector ingestion

### Extracted text checks

- Text is nonempty and page/text volume is plausible.
- Character quality, replacement/control characters, whitespace, and repeated headers/footers are within expected bounds.
- Flag pages with unusually little text or abrupt changes in text volume.
- Manually inspect samples against rendered pages, especially columns, tables, and equations.
- Treat heuristics as flags, not universal rejection rules; math- and table-heavy content differs from prose.

### Chunk checks

- No empty/whitespace-only chunks; ordinals are sequential and section metadata is present.
- Token counts are plausible; enforce the actual model limit with its tokenizer.
- Review unusually small/large chunks, exact duplicates, and missing pre-heading text.
- Check expected sections or approximate coverage of extracted body text.
- Compare overlapping chunks with their positions before classifying repetition as an error.

### Embedding checks

- Number of vectors equals number of chunks, with correct order and one-to-one mapping.
- Dimensions match the configured index; all values are finite and vectors are not unexpectedly zero.
- Check norms if embeddings are expected to be normalized.
- Investigate exact/near-identical vectors alongside their input texts. Repeated text may be legitimate; different texts yielding identical vectors can indicate a batching/mapping/embedder issue.

### Persistence and retry behavior

Embed and validate the complete set before replacing old chunks. If 98 of 100 embeddings succeed, fail/retry the ingestion and retain the old indexed set rather than silently publishing an incomplete paper. `replace_chunks()` performs replacement transactionally, but embedding and validation should happen before that call. Record failure details and keep retries idempotent.

## 7. Intentional overlap versus accidental duplicates

The current overlap copies whole trailing sentences into the next window up to an approximate token target. Expected overlap is therefore between adjacent chunks in the same section. Identical whole chunks, unexpectedly large overlap, or repeated content in nonadjacent/unrelated sections are stronger duplicate signals. Check text and positions before treating identical vectors as a bug.

Tune overlap with a representative retrieval evaluation. Measure recall/MRR/nDCG and boundary-spanning evidence recovery alongside chunks per paper, embedding/index cost, and duplicate retrieved evidence. Keep overlap only where retrieval or downstream evidence quality improves enough to justify added cost.

## 8. Likely interviewer follow-ups

1. What happens when a single sentence exceeds the chunk size, and how would you guarantee the embedder limit?
2. What text may be lost if some headings are detected but there is a preamble before the first heading?
3. How do you separate table rows from numbered headings after layout is flattened?
4. How do you verify reading order in multi-column PDFs?
5. What metadata belongs on a chunk for citations, filtering, and debugging?
6. How would you represent a table so retrieval preserves header/value relationships?
7. If only 98 of 100 embeddings succeed, what happens to existing indexed data?
8. How would you diagnose identical vectors: duplicated text, overlap, ordering bug, or embedder issue?
9. How would you tune overlap and chunk size using retrieval metrics?
10. Which checks are hard failures and which should only flag a document for review?

## 9. Compact answer structure

For each interview response, use this order:

1. State what the current code does.
2. Name its limitation or failure mode.
3. Propose a targeted check or change.
4. Explain how you would evaluate it.
5. Describe the failure policy (preserve old data, retry, or flag) where relevant.

## 10. Evaluation metrics: retrieval, context relevance, and faithfulness

Keep retrieval quality, context quality, and answer quality as separate measurements. A single aggregate score can hide whether a failure came from retrieval, distracting context, or generation.

### Recall@K

For one query:

`Recall@K = number of relevant chunks in the top K / total number of relevant chunks for the query`

Compute the ratio per query, then report the mean across queries (and uncertainty, such as a bootstrap confidence interval). Define relevance before evaluation: for example, chunks that contain evidence needed to answer the query. Build a representative set of questions, have domain-informed annotators label relevant evidence, and preserve the judgments and corpus/chunker version when comparing systems. If chunk boundaries change, remap or re-judge evidence against the new chunks rather than assuming old chunk IDs still apply.

Recall@K measures how much labeled relevant evidence was retrieved. It does not measure ranking quality within the top K, whether retrieved evidence is sufficient as a set, whether irrelevant context distracts generation, or whether the final answer is correct. Pair it with a rank-sensitive measure such as MRR or nDCG, and with context/answer measures.

### Questions whose evidence spans chunks

If a query needs chunks A and B together, annotate a minimal sufficient evidence group `{A, B}`. Do not treat finding A alone as a complete success merely because A is relevant. Report ordinary Recall@K to show partial evidence recovery, and also:

`Complete Evidence@K = queries where top K contains every chunk in at least one sufficient evidence group / total queries`

Some questions may have multiple valid evidence paths. In that case, annotate multiple sufficient groups and count a query as complete when retrieval contains all members of any one group. Store chunk adjacency/overlap as structural metadata if useful, but do not interpret adjacency as proof that both chunks are required for a particular answer. Requirement is query-specific and should come from evidence annotation.

### Context Relevance

Context relevance asks whether the retrieved context is useful for answering the query. Use an explicit, task-specific rubric and anchor it with examples. One possible per-chunk rubric is:

- `0`: unrelated; provides no useful evidence.
- `1`: topically related, but provides no usable answer evidence.
- `2`: supports part of the answer or provides one necessary piece of split evidence.
- `3`: directly supports a substantial part of the answer.

For K retrieved chunks:

`Context Relevance@K = sum of chunk scores / (3 * K)`

This yields a score from 0 to 1. State whether chunks are judged independently or in the context of the other retrieved chunks; keep the rule fixed. The rubric is not universal: sample real queries and contexts, have multiple reviewers label examples independently, resolve disagreements, and retain agreed examples as anchors. Report reviewer agreement and inspect disagreements. An LLM judge can scale scoring only after validation against human labels; keep its prompt and model version fixed and audit samples.

An average can hide clutter: one excellent chunk can be diluted by irrelevant chunks, or several mediocre scores can mask a missing critical fact. Report useful-context coverage or distraction rates alongside the mean, and compare the scores with answer outcomes.

### Answer Faithfulness

Faithfulness asks whether claims in the generated answer are supported by the retrieved context. Break the answer into factual claims, then label each claim as supported, contradicted, or not supported by the context. A simple claim-level score is:

`Faithfulness = supported answer claims / all answer claims`

Also report unsupported and contradicted claim rates separately, since a wrong claim contradicted by evidence is more serious than a claim with no evidence. Define treatment of unverifiable claims, citations, and non-factual language in the rubric. Evaluate correctness against reference answers separately: an answer may faithfully repeat incomplete context and still be wrong, while a correct answer may be unsupported by the retrieved context.

For reproducibility, use a fixed claim extraction and judging rubric, double-label a representative sample, measure reviewer agreement, and validate any automated judge against those labels. Keep retrieval metrics, context relevance, answer correctness, and faithfulness side by side so regressions can be diagnosed.

### Evaluation design and reporting

- Stratify questions by evidence type: direct fact, multi-part answer, table/equation, rare exact term, and boundary-spanning evidence.
- Freeze corpus snapshot, chunker configuration, query set, relevance rubric, and judge versions for system comparisons.
- Report metric means with uncertainty and break results down by query type; include examples of failures.
- Avoid treating an unlabeled candidate as certainly irrelevant. Incomplete relevance pools make recall estimates artificially low.
- Track cost/latency and context token use alongside quality so larger K or overlap is not considered free.

## 11. BM25F: interpreting the formula

The BM25F formulas in [math.md](math.md) can be read as three steps: normalize term frequency for each field's length, weight fields by importance, then apply diminishing returns to repeated occurrences.

For term `t` in field `f`:

`length-adjusted tf = tf(t, f) / (1 - b_f + b_f * len_f / avglen_f)`

Here `len_f / avglen_f` compares the field's length with the average length for that field. `b_f` is the field-specific length-normalization strength: it controls how much the field's term frequency is adjusted based on its length relative to the average. It is normally between 0 and 1. At `b_f = 0`, there is no length adjustment because the denominator is always 1. At `b_f = 1`, there is full adjustment by the length ratio. Values between 0 and 1 give partial adjustment. At average length, the denominator is 1 regardless of `b_f`. For a field twice average length and `b_f = 0.75`, the denominator is `1 - 0.75 + 0.75 * 2 = 1.75`, so a raw term frequency of 4 becomes about 2.29. BM25F can use a different `b_f` for each field because length has different meaning in a title and an abstract; the guide's settings use `b_title = 0.3` and `b_abstract = 0.75`.

`w_f` is the field-importance multiplier. For example, `w_title = 2` and `w_abstract = 1` makes a title occurrence count twice as strongly as an abstract occurrence in the combined weighted frequency, before saturation. These are tuning parameters, not probabilities; select them on validation labels and report ranking metrics.

After length adjustment and field weighting, BM25F combines a term's contribution using the saturation function:

`g(x) = x * (k1 + 1) / (x + k1)`

This is the term-frequency factor in BM25's score, with inverse document frequency multiplied outside it. `g(0) = 0`, `g(1) = 1`, and as `x` grows it approaches `k1 + 1`. Repetitions help, but with diminishing increments. `k1` controls how quickly saturation happens. The expression is a chosen BM25 model shape; it is not derived from the PDF chunker.

In BM25F, fields are combined before saturation. This avoids separately saturating title and abstract hits and then awarding two near-maximum rewards. The field weights and length-normalization settings should reflect field behavior and be tuned against judged retrieval data.
