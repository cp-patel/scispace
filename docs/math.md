# The math guide

Every algorithm in this project, explained the same way:

1. **The idea** in plain words, with an everyday comparison.
2. **The formula**, written the way the code computes it.
3. **Why it works**: a short derivation or proof sketch.
4. **Where it lives**: the code and the test that proves it.
5. **Interview questions** with answers.

Sections 1 to 6 and 10 are **core**: know them cold. The rest are **bonus**.

## Results at a glance

From the selection eval (`evals/results/selection-2026-10-15.md`): 7 published surveys from 6 fields. For each one, the pipeline ran as if on the survey's publication date, and the survey's own reference list served as the answer key. Values are means over surveys, with 95% bootstrap intervals in brackets.

| Ranking | Recall@150 | nDCG@100 |
|---|---|---|
| OpenAlex's own order | 0.065 [0.032, 0.097] | 0.067 [0.035, 0.097] |
| BM25F only | 0.080 [0.035, 0.131] | 0.054 [0.024, 0.093] |
| Dense vectors only | 0.203 [0.177, 0.235] | 0.165 [0.123, 0.208] |
| RRF of BM25F + dense | 0.186 [0.167, 0.212] | 0.129 [0.094, 0.167] |
| ... + Personalized PageRank | 0.242 [0.196, 0.303] | 0.180 [0.131, 0.236] |
| ... + Rocchio (full system) | 0.245 [0.192, 0.319] | 0.184 [0.134, 0.238] |

| Comparison (nDCG@100) | Mean difference | p-value |
|---|---|---|
| RRF of text signals vs BM25F only | +0.074 | 0.016 |
| adding PageRank | +0.051 | 0.032 |
| adding Rocchio | +0.004 | 0.45 |

Four honest readings:
- **Biggest gains.** The citation graph (PageRank) and the vectors are the big wins. Both gains are statistically real.
- **A weak signal can hurt.** Equal-weight fusion let a weak keyword signal pull the vectors down (dense alone beats the BM25F + dense fusion). That is a reason to tune or learn the fusion weights.
- **Rocchio.** Its gain here is too small to tell from luck.
- **Absolute recall is low.** The offline planner writes narrow keyword queries, so many cited papers never enter the pool. An LLM planner is the first fix to try.

## 1. BM25F (core)

**The idea.** Count how often the query's words appear in a paper, with three corrections. Rare words count more than common ones. The tenth repeat of a word adds far less than the first. A long abstract does not win just by being long. Think of grading an essay on photosynthesis: the first mention shows the essay is on topic, the twentieth mention adds nothing, and mentions of "the" mean nothing at all.

**The formula** (`ranking/bm25.py`). For a paper `d` and query terms `t`:

```
tf~(t, d) = sum over fields f of  w_f * tf(t, f) / (1 - b_f + b_f * len_f / avglen_f)
score(d)  = sum over t of  IDF(t) * tf~(t, d) * (k1 + 1) / (tf~(t, d) + k1)
IDF(t)    = ln( (N - n_t + 0.5) / (n_t + 0.5) + 1 )
```

Defaults: `k1 = 1.2`; title `w = 2.0, b = 0.3`; abstract `w = 1.0, b = 0.75`. `N` is the pool size and `n_t` the number of pool papers containing `t`.

**Why it works.**
- **Saturation.** `g(x) = x (k1 + 1) / (x + k1)` is 0 at 0 and 1 at 1, and climbs toward `k1 + 1` but never reaches it. Its slope `k1 (k1 + 1) / (x + k1)^2` keeps shrinking, so each repeat adds less than the one before. Each term's contribution is therefore below `IDF(t) * (k1 + 1)`. This shape approximates the "2-Poisson" model, in which a word is either central to a document or appears by chance.
- **The `+ 1` inside the log.** Without it, a word found in more than half the papers would get a negative weight and *lower* the score. With it, the weight is never negative.
- **Fields are combined before saturation.** If each field were saturated separately and then added, a word found in both the title and the abstract would collect two near-full rewards. BM25F treats the paper as one weighted bag of words.
- **IDF over the topic's own pool.** Words shared by the whole topic ("retrieval" in a retrieval pool) get a low IDF, which is right: they do not separate good candidates from weak ones.

**Where it lives.** `ranking/bm25.py`, used by `funnel.text_signals` (papers) and `writing.rank_chunks` (passages). The tests in `tests/ranking/test_bm25.py` check saturation, the upper bound, rarity, length normalization, and that a title hit beats an abstract hit.

**Interview questions.**
- *Why BM25 rather than TF-IDF?* TF-IDF grows linearly with the count and ignores length, so long, repetitive texts win. BM25 saturates the count and normalizes for length, and it has two knobs to tune.
- *When does keyword search beat embeddings?* With rare, exact terms: gene names, model names like "LoRA", chemical formulas, acronyms, and version numbers. Embeddings blur these into nearby meanings. In our eval, BM25F alone was weak because the offline planner's keywords were generic; with precise keywords it does better. That is exactly why the system keeps both signals.
- *What do `k1` and `b` do?* `k1` sets how fast repeats stop helping (near 0: only presence counts; large: close to raw counts). `b` sets how hard length is punished (0: not at all; 1: fully).

## 2. Cosine similarity and Rocchio feedback (core)

**The idea.** An embedding turns a text into a vector, an arrow in 384 dimensions, so that texts about the same thing point in similar directions. Cosine similarity measures how closely two arrows point the same way: 1 for the same direction, 0 for unrelated. Rocchio feedback then nudges the query's arrow toward the arrows of its best early results, like adjusting your aim after seeing where your first few darts landed.

**The formula** (`ranking/dense.py`).

```
cos(a, b) = a . b / (|a| |b|)        and for unit vectors simply  a . b
q'        = alpha * q + beta * mean(top R results) - gamma * mean(non-relevant), then scaled to length 1
```

Defaults: `alpha = 1`, `beta = 0.75`, `gamma = 0`, and R = the top 20 papers by the facet's first-pass score.

**Why it works.**
- **Unit vectors.** Once every vector has length 1, cosine is a dot product, so scoring a whole pool against all facets is one matrix multiply, `E @ Q.T`.
- **Rocchio's direction.** Rocchio (1971) asks for the query that is most similar to the relevant centroid and least similar to the non-relevant one. In the vector model, the answer points along `mean(relevant) - mean(non-relevant)`; the `alpha * q` term keeps the user's original intent in the mix.
- **Pseudo-relevance feedback.** Using the top results as "relevant" without anyone checking them is called pseudo-relevance feedback. The risk is *query drift*: if the early results are off topic, the query moves the wrong way.
- **Why `gamma = 0`.** The bottom of a shortlist is not truly irrelevant (those are near misses), so subtracting it can hurt. Real negatives, such as papers the judge rejected, could justify `gamma > 0`.
- **Asymmetric embedding.** bge models embed queries with an instruction prefix and documents without one; `embed.py` exposes both (`embed_queries`, `embed_docs`).

**Where it lives.** `ranking/dense.py` and `funnel.compute_signals`; `test_rocchio_moves_query_toward_relevant_centroid` checks the direction over 20 random seeds.

**Interview questions.**
- *Why normalize vectors?* Cosine becomes a dot product, so one matrix multiply scores everything, and long texts are not favored by vector length.
- *What can go wrong with pseudo-relevance feedback?* Query drift. Keep `alpha` dominant, keep R small, and measure: here Rocchio added +0.004 nDCG with p = 0.45, so we cannot claim it helps on this eval.
- *Why did dense vectors beat keywords 2.5 to 1 on recall?* Surveys cite papers that describe the same idea in different words; vectors capture meaning, while keywords need the exact words.

## 3. Personalized PageRank (core)

**The idea.** Picture a reader who starts at one of the strongest papers for the topic and keeps following its references. Each step, with probability 15%, they jump back to a strong paper. The papers this reader keeps landing on are the ones the topic builds on, even when they use different words.

**The formula** (`ranking/pagerank.py`).

```
pi = (1 - alpha) * v + alpha * P^T pi
```

`P[i, j] = 1 / outdeg(i)` when paper `i` cites paper `j` (both in the pool). `v` is the teleport vector: the top 50 papers by text score, weighted by that score, summing to 1. Papers with no in-pool references ("dangling") send their mass back to `v`. Defaults: `alpha = 0.85`, stop when the L1 change is below `1e-10`.

**Why it works.**
- **It stays a probability distribution.** Non-dangling mass flows along edges, and dangling mass returns through `v`, so the total stays exactly 1.
- **It converges, to one answer.** Write one step as `T(x) = (1 - alpha) v + alpha M x`, where `M = P^T + v d^T` has non-negative columns that each sum to 1. For any two vectors, `|T(x) - T(y)|_1 = alpha |M (x - y)|_1 <= alpha |x - y|_1`. Each step shrinks the error by at least a factor of `alpha`, which is a contraction, so there is exactly one fixed point, and power iteration reaches it. With `alpha = 0.85`, about 140 steps get the error below `1e-10` (`0.85^142` is about `1e-10`).
- **There is a closed form.** `pi = (1 - alpha) (I - alpha M)^(-1) v`. The test checks the iteration against this direct solve.
- **Why "personalized" beats raw citation counts.** Global citation counts reward famous papers from any field. Teleporting only to topic-relevant seeds makes importance topic-specific. It also adds evidence that does not depend on wording at all.

**Where it lives.** `ranking/pagerank.py`; `funnel.compute_signals` builds the edges from each candidate's reference list. `test_pagerank_sums_to_one_and_matches_direct_solve` runs over 15 random graphs.

**Our numbers.** Adding PageRank to the text signals raised recall@150 from 0.186 to 0.242 and nDCG@100 by +0.051 (p = 0.032): the largest statistically supported gain in the eval.

**Interview questions.**
- *What does `alpha` control?* How far the reader wanders before jumping back. A high `alpha` gives longer, more global walks; a low one stays near the seeds. It also sets the convergence speed (error shrinks like `alpha^t`).
- *How do you handle dangling nodes?* Send their mass to the teleport vector, which keeps a valid distribution and the contraction argument intact.
- *How does it scale?* Each step costs O(edges) with sparse arrays. At 200 million papers, precompute the global graph and warm-start each personalized run from a cached solution.

## 4. Reciprocal Rank Fusion (core)

**The idea.** Several judges each rank the contestants. Instead of comparing their raw scores (which use different scales), give each contestant `1 / (60 + place)` points from each judge and add them up. A paper ranked well by several judges beats one ranked first by a single judge.

**The formula** (`ranking/fusion.py`).

```
RRF(d) = sum over lists i of  w_i / (k + rank_i(d)),    k = 60,  absent from a list: adds 0
```

Lists used: BM25F and dense for every facet, each with weight `1 / F`, plus PageRank with weight 1, so text and graph get comparable say.

**Why it works.**
- **Only ranks matter.** Any increasing change to a list's scores (multiply by 10 and add 3, take the exponential) leaves its ranks, and therefore RRF, unchanged. `test_rrf_only_depends_on_ranks_not_score_scale` checks this. So BM25 points, cosines and PageRank mass, which live on unrelated scales, can be combined without calibration.
- **`k = 60` flattens the top.** Rank 1 versus rank 2 differ by only `1/61 - 1/62`, about 0.0003, so small disagreements at the top do not dominate, and agreement across lists wins. Cormack, Clarke and Buettcher (2009) found 60 robust across collections.
- **The honest limit.** Equal weights let a weak list dilute a strong one. In our eval, dense alone (nDCG 0.165) beat the BM25F + dense fusion (0.129), because BM25F was weak with the offline planner's keywords. The fix is to tune the weights, or learn them from labels.

**Interview questions.**
- *Why not average normalized scores?* Score shapes differ: BM25 is unbounded and heavy-tailed, while cosines bunch together in a narrow band. Min-max normalization lets one outlier squash everything else. RRF avoids the problem entirely.
- *What does RRF throw away?* How big the gaps are: a runaway winner and a narrow winner count the same.
- *How would you learn the weights?* Use the survey labels: logistic regression or a learning-to-rank model on per-list features, validated by leaving one survey out at a time.

## 5. Bayes' rule for the LLM judge (core)

**The idea.** Treat the judge like a medical test with known error rates. A "yes" from a test that rarely gives false alarms is strong evidence; a "yes" from a test that says yes to almost everything is weak. And a positive result means more for something that was already likely.

**The formula** (`ranking/bayes.py`).

```
sensitivity Se = P(judge says include | paper is relevant)
specificity Sp = P(judge says exclude | paper is not relevant)
LR+ = Se / (1 - Sp)            LR- = (1 - Se) / Sp
posterior odds = prior odds * LR        p = odds / (1 + odds)
```

The prior comes from the calibration table: the share of relevant papers at each shortlist rank, measured by the eval. Before that table exists, a straight line from 0.6 at rank 1 to 0.1 at the last rank stands in.

**Why it works.**
- **Odds form.** Bayes' rule says `P(R | yes) / P(not R | yes) = [P(yes | R) / P(yes | not R)] * [P(R) / P(not R)]`. That is the odds form above.
- **An uninformative judge changes nothing.** If `Se = 1 - Sp`, then `LR = 1` and the posterior equals the prior. That is the right answer for a coin flip, and a test checks it.
- **Clipping.** Probabilities and rates are clipped to `[1e-6, 1 - 1e-6]`, so the update never divides by zero.

**Our numbers.**
- **The offline judge is weak.** Its sensitivity is 0.69 and specificity 0.46. That gives LR+ of about 1.28 and LR- of about 0.67, and Cohen's kappa of 0.06, barely above chance. The math correctly gives it little weight.
- **The calibration table is flat.** It shows 0.11 (ranks 1-10), 0.14 (11-25), 0.15 (26-50), 0.17 (51-100) and 0.08 (101+). So the straight-line fallback (0.6 at the top) was about five times too confident. Measuring beats assuming.

**Interview questions.**
- *Why not trust the LLM's yes or no?* Because it has error rates. Combining its verdict with retrieval evidence beats either alone, and Bayes says exactly how much weight each deserves.
- *How do you get sensitivity and specificity without human labels?* Use survey references as labels. Negatives are noisy, because a survey does not cite every good paper, so measured specificity is a floor. Add a small hand-labeled set to check.
- *Can you ask the LLM 5 times and multiply the likelihood ratios?* Only if the errors were independent. Samples from one model share its blind spots, so multiplying overstates confidence. We use one verdict per paper.

## 6. Submodular selection and lazy greedy (core)

**The idea.** Choosing 40 papers is a set problem, not a ranking problem: the 40 top-scored papers are often 40 near-copies of one idea. We want each pick to add something new. Like picking a sports team, a second goalkeeper adds less than the first, so the value of a player depends on who is already chosen. This "diminishing returns" property is called **submodularity**.

**The formula** (`ranking/select.py`).

```
F(S) = sum over facets f of  w_f * log(1 + sum over d in S of  p_d * r_{d,f})
     + (mu / n) * sum over candidates i of  p_i * max over j in S of  s_{i,j}
```

`p_d` is the paper's probability of relevance (from section 5). `r_{d,f}` is its relevance to facet `f`, between 0 and 1. `s` is cosine similarity clipped at 0. The first term balances the facets: each facet's value grows like a log, so piling papers onto one facet stops paying. The second term is **facility location**: every candidate wants a close "representative" in the set, which spreads the picks across the whole shortlist.

**Why it works.**
- **Definitions.** `F` is *monotone* if adding a paper never lowers it. It is *submodular* if a paper's gain shrinks as the set grows: `F(A + x) - F(A) >= F(B + x) - F(B)` whenever A is inside B.
- **Both terms qualify.** A concave function (the log) of a non-negative sum is monotone submodular. Facility location with non-negative similarities is too: a paper's gain, `sum_i w_i * max(0, s_{i,x} - best_i)`, can only fall as each `best_i` rises. Sums of such functions keep both properties.
- **The greedy guarantee: at least (1 - 1/e) of the best, about 63%.** Let OPT be the best set of size k, with value F*, and let S_i be greedy's set after i steps.
  - Monotonicity and submodularity give `F* <= F(S_i) + sum over o in OPT of gain(o | S_i) <= F(S_i) + k * (F(S_(i+1)) - F(S_i))`, because greedy took the largest gain.
  - So the remaining gap `g_i = F* - F(S_i)` shrinks: `g_(i+1) <= (1 - 1/k) g_i`.
  - After k steps, `g_k <= (1 - 1/k)^k F* <= F*/e`, so `F(S_k) >= (1 - 1/e) F*` (Nemhauser, Wolsey, Fisher 1978).
- **Lazy greedy picks the same set with much less work.** Gains only shrink as the set grows, so a gain computed earlier is an upper bound on today's. Keep all candidates in a max-heap of old gains, pop the top, and recompute its gain. If it still beats every other upper bound, it is the true best, and nobody else needs rechecking (Minoux 1978). `test_lazy_greedy_picks_exactly_what_plain_greedy_picks` checks 25 instances.
- **The budgeted version** (fit passages into a token budget). Greedy on gain-per-token alone can fail badly: it may prefer many cheap, mediocre chunks over one great one. Taking the better of that greedy result and the best single chunk that fits is proven to reach `1/2 (1 - 1/e)` of the optimum.

**Our numbers.** Recall@40 was 0.132 for submodular versus 0.144 for plain top-k. The difference of -0.012 has p = 1.0, so there's no evidence of a difference either way. Every method covered all facets.

The honest reading: a survey's reference list rewards finding the papers it cites, not variety, so this eval cannot see what diversity buys. Measuring coverage properly needs labels per sub-topic, or a human judge.

**Where it lives.** `ranking/select.py`, `funnel.select`, `writing.pick_within_budget`. The tests in `tests/ranking/test_select_dpp.py` check:
- greedy against brute force on small cases (the `1 - 1/e` bound)
- submodularity on random sets
- lazy greedy equals plain greedy
- the budget is never exceeded, and the budgeted bound holds

**Interview questions.**
- *Why is selection a set problem?* Because a paper's value depends on what else is in the set. Top-k by score repeats itself.
- *Sketch the 1 - 1/e proof.* (Above: the gap to the optimum shrinks by a factor of `1 - 1/k` every step.)
- *Why can lazy greedy skip work?* Submodularity makes old gains upper bounds, so the top of the heap, once refreshed, is provably the best.

## 7. MMR (bonus)

**The idea.** Maximal Marginal Relevance: repeatedly add the paper with the best trade-off between being relevant and being different from what is already chosen.

**The formula.** `next = argmax over d of [ lambda * rel(d) - (1 - lambda) * max over s in S of sim(d, s) ]`, with `lambda = 0.7`.

**Why it is only a baseline.** MMR is a rule, not the maximizer of any fixed set objective, so there is no guarantee about how good its set is. It is simple and fast, which is why it is the standard baseline. In our eval: recall@40 0.141, mean pairwise cosine 0.814.

**Interview question.** *MMR or submodular?* MMR when you need something quick and explainable; a submodular objective when you want to state what "good set" means (coverage, balance) and get a guarantee.

## 8. Determinantal Point Process, DPP (bonus)

**The idea.** Score a whole set at once by the volume its vectors span. Picking photos for an album: you want good photos (quality) that are not the same shot twice (diversity). A set of near-duplicates spans almost no volume, while a set of strong, different items spans a lot.

**The formula** (`ranking/dpp.py`).

```
L = diag(q) K diag(q) + eps * I,   K = E E^T (cosine Gram matrix),   q_i = exp(theta * p_i),  theta = 2
P(S) is proportional to det(L_S)
```

**Why it works.**
- **Determinant as volume.** `det(L_S) = prod over i in S of q_i^2 * det(E_S E_S^T)`. The Gram determinant `det(E_S E_S^T)` is the squared volume of the parallelepiped spanned by the vectors in S. Quality and diversity are multiplied into one number.
- **Greedy by Cholesky, in O(N k^2).** Write `L_S = C C^T` (its Cholesky factor).
  - Adding item j multiplies the determinant by `d_j^2 = L_jj - |c_j|^2`. That is the part of j's vector not already covered by the chosen items (a Schur complement).
  - Keep `c_i` and `d_i^2` for every candidate. After picking j, compute one new Cholesky entry for every i, `e_i = (L_ji - <c_j, c_i>) / d_j`, append it to `c_i`, and subtract: `d_i^2 -= e_i^2`. Each step costs O(N k).
  - `log det(L_S)` is the sum of `log d_j^2` at each pick. `test_dpp_log_det_matches_numpy` checks it against NumPy.
- **No `1 - 1/e` here.** `log det` is submodular but **not monotone**: an item with `d_j^2 < 1` lowers it. So the greedy guarantee from section 6 does not carry over. DPP greedy is a strong heuristic.
- **A size limit.** `K` has rank at most 384 (the embedding size), so no more than 384 items can have a positive determinant. That's fine for 40.

**Our numbers.** DPP picked the most varied sets: mean pairwise cosine 0.779, versus 0.825 for top-k. Its recall@40 was 0.130, versus 0.144 for top-k; the difference has p = 0.57, so it could easily be luck.

**Interview questions.**
- *Why `exp(theta * p)` for quality?* It makes quality enter the determinant multiplicatively. `theta` sets the trade-off: a higher `theta` favors relevance, a lower one favors diversity.
- *DPP or submodular?* DPP gives one principled score for a whole set and favors sets that point in many directions. The submodular objective lets you encode facets explicitly and comes with a guarantee.

## 9. Spectral clustering and the eigengap (bonus)

**The idea.** Turn the 40 chosen papers into a graph where similar papers are linked. Groups that are tightly linked inside and loosely linked to each other become the review's sections. Think of friend groups at a school: many ties inside each group, few between groups.

**The formula** (`ranking/cluster.py`).

```
W = beta * max(0, cos) + (1 - beta) * coupling      beta = 0.7; keep each paper's 8 strongest links; symmetrize
L_sym = I - D^(-1/2) W D^(-1/2)                      D = diagonal of row sums of W
k = argmax over k in [3, 8] of (lambda_(k+1) - lambda_k)
```

Then take the eigenvectors of the k smallest eigenvalues, scale each row to length 1, and run k-means (k-means++ starts, 10 restarts). **Bibliographic coupling** is the cosine between two papers' reference lists: papers that cite the same work are related.

**Why it works.**
- **Zero eigenvalues count disconnected groups.** If the graph has c separate pieces, `L_sym` has the eigenvalue 0 exactly c times. The matching eigenvectors are constant within each piece.
- **Nearly separate groups give nearly zero eigenvalues.** You get c small values, then a jump. The **eigengap** reads off c.
- **Why normalize the rows.** Normalizing each row (the Ng, Jordan, Weiss method) places papers from the same group near the same point on a sphere, where k-means separates them easily.
- **The link to graph cuts.** The eigenvectors are the relaxed solution to the normalized-cut problem: split the graph cutting as few weighted links as possible, relative to group size.

**In the sample run** the smallest eigenvalues were 0.000, 0.264, 0.568, 0.680, 0.702, 0.723. Among k from 3 to 8, the largest jump comes after the third (0.680 - 0.568 = 0.112), so the review got 3 sections.

**Where it lives.** `ranking/cluster.py`, `agent/nodes.py` (`outline`). `test_spectral_recovers_planted_groups_and_their_count` plants 3 groups and checks that both the groups and k = 3 are found.

**Interview questions.**
- *Why not k-means on the embeddings directly?* It needs k in advance, assumes round clusters, and ignores citations. The graph uses both text and citations, and the eigengap picks k.
- *What if there is no clear gap?* Then the data has no strong group structure. Fall back to a fixed k or a silhouette score, and say so in the report.

## 10. Metrics and statistics (core)

**Recall@k, capped.** `hits in top k / min(k, size of the answer key)`. Capping means a perfect top 40 scores 1 even when the survey cites 213 papers.

**nDCG@k (binary).** `DCG = sum over hits at positions i of 1 / log2(i + 1)`, divided by the best possible DCG. Hits near the top count more.

**MRR.** `1 / position of the first hit`.

**The bootstrap.** We have 7 survey scores and want a range for the true mean. Resample the 7 scores with replacement 10,000 times, take each resample's mean, and keep the middle 95% of those means. That is the confidence interval. `test_bootstrap_ci_covers_true_mean_about_95_percent_of_the_time` checks the coverage.

**The paired permutation (sign-flip) test.** To ask whether A beats B, take the 7 per-survey differences A - B. If A and B were equally good, each difference would be equally likely to be positive or negative. Flip signs at random 10,000 times; the p-value is the share of flips whose mean difference is at least as large as the real one. Pairing matters: some surveys are easy and some hard, and comparing within each survey cancels that out.

**Cohen's kappa.** `(observed agreement - chance agreement) / (1 - chance agreement)`: 0 means no better than chance, 1 means perfect.

**Caveats.**
- With 7 surveys, the intervals are wide.
- Survey references are an incomplete answer key, so recall is the trustworthy number and precision is only a floor.
- OpenAlex's own ordering already mixes in citation counts.

**Interview questions.**
- *Why a survey as the answer key?* It is an expert-curated selection for a real topic, available for free in many fields, and its date tells you what the pipeline was allowed to know.
- *Why is a paired test better here?* Survey difficulty varies a lot; pairing removes that variation and leaves only the method difference.

## 11. Locality-sensitive hashing, LSH (bonus)

**The idea.** Slice the space with random flat cuts (hyperplanes) and record which side of each cut a vector falls on. Similar vectors land on the same side of most cuts, so they share a bucket. Search only looks in the query's buckets.

**Why one bit agrees with probability `1 - theta / pi`.** Take two vectors at angle `theta` and a random direction `r` from a Gaussian. Only the part of `r` in the plane of the two vectors matters, and by symmetry its angle is uniform around the circle. The cut separates the two vectors exactly when that angle falls inside one of two arcs of size `theta`. That happens with probability `2 theta / (2 pi) = theta / pi` (Charikar, 2002). `test_lsh_bit_collision_rate_matches_one_minus_theta_over_pi` checks it with 1,024 random cuts.

**AND and OR.** A bucket key uses b bits, so near vectors share it with probability `p^b`. That's AND: strict, and it cuts false matches. With L independent tables, a true neighbor is found with probability `1 - (1 - p^b)^L`. That's OR: it rescues recall. Together they form an S-shaped curve you tune with b and L.

**Benchmark (synthetic, 3,000 vectors).** 8 bits and 8 tables: recall@10 0.67. 8 bits and 16 tables: 0.91. 12 bits and 32 tables: 0.77, because more bits make buckets too strict.

## 12. IVF, inverted file index (bonus)

**The idea.** Group the vectors into cells with k-means (`nlist` cells). A search looks only inside the `nprobe` cells nearest the query, then scores those vectors exactly. It is like checking only the library shelves for the right subject.

**The trade-off.** A larger `nprobe` means higher recall and slower search. Benchmark (synthetic, 64 cells): `nprobe = 1` gives recall 0.94, `nprobe = 8` gives 1.0. Neighbors near a cell boundary are the misses at small `nprobe`.

## 13. Product quantization, PQ (bonus)

**The idea.** Compress each vector to a few bytes.
- **Encode.** Cut the 384 numbers into m pieces. For each piece, learn 256 typical values with k-means. Store only the id (one byte) of the nearest typical value per piece. A 1,536-byte vector becomes m bytes: 48 bytes is 32 times smaller.
- **Score.** For a query, precompute its inner product with all 256 typical values of every piece: a small `m x 256` lookup table. A database vector's approximate score is then m table lookups added up. This is "asymmetric distance": the query stays exact, only the database is compressed.

**Why the lookup works.** The inner product splits over pieces: `<q, x> = sum over pieces s of <q_s, x_s>`, and each `x_s` is replaced by its typical value.

**Re-ranking.** PQ alone loses detail. Benchmark (synthetic): PQ alone reached recall 0.40 at m = 48. Scoring the top 100 PQ candidates again with the full vectors brought recall to 1.0. Production systems pair compressed search with a small exact re-rank.

## 14. HNSW, hierarchical navigable small world graphs (bonus)

**The idea.** Link each vector to a few close neighbors, forming a graph that you can walk toward any query. Add sparser "express lane" layers on top, like a subway with local and express lines. Start on the top layer, hop greedily toward the query, drop down a layer, and repeat. Finish with a careful search on the bottom layer that keeps the `ef` best candidates.

**Why the layers thin out by a factor of M.** Each node draws its top layer as `floor(-ln(U) * mL)`, where U is uniform on (0, 1] and `mL = 1 / ln(M)`. Then `P(level >= l) = P(U <= e^(-l / mL)) = M^(-l)`. Each layer up keeps about 1/M of the nodes, so a search needs about `log_M(N)` layers, much like a skip list.

**The neighbor heuristic.** When linking a new node, keep a candidate only if it is closer to the new node than to any neighbor already kept. Links then point in different directions instead of into one dense clump, which keeps the graph navigable between clusters.

**The knobs.** `M` (links per node) trades memory for recall. `ef_construction` trades build time for graph quality. `ef_search` trades query speed for recall.

**Benchmark (synthetic, 3,000 vectors).**
- **Our HNSW:** recall 1.0 at 11,454 queries per second with `ef = 16`.
- **pgvector's HNSW:** recall 1.0 at 3,888 queries per second. Its number includes a SQL round trip for every query.
- **Build:** pgvector, written in C, built its index in 0.44 seconds; our pure-Python index took 2.9 seconds.
- **Scale caveat:** at this size, exact search (31,897 queries per second) beats every index. Indexes pay off as N grows, and the real-data run on 50,000 abstracts (`lab/bench.py --source openalex`) is where that shows.

**Interview questions.**
- *HNSW or IVF-PQ?* HNSW gives the best recall per query time but stores full vectors plus links (memory-heavy). IVF-PQ is far smaller and fits billions of vectors on less hardware, at some recall cost, recovered by re-ranking.
- *Why did exact search win the small benchmark?* At a few thousand vectors, one matrix multiply in optimized code is faster than any index's overhead. Do not add an index until measurements say you need one.
- *How do filters interact with HNSW?* The index finds nearest vectors first, then a filter can discard most of them. pgvector's iterative scans keep searching until enough rows pass.
