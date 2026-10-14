# Selection eval

Surveys: 7. Values are means over surveys with 95% bootstrap intervals.

## Ranking

| Ranking | Recall@150 | nDCG@100 |
|---|---|---|
| openalex_order | 0.0647 [0.0319, 0.0967] | 0.0666 [0.0351, 0.0972] |
| bm25f | 0.08 [0.0348, 0.1309] | 0.0541 [0.0238, 0.0932] |
| dense | 0.2033 [0.1771, 0.2346] | 0.1654 [0.1233, 0.2083] |
| rrf_text | 0.1857 [0.1667, 0.212] | 0.1285 [0.0942, 0.1666] |
| rrf_text_ppr | 0.2416 [0.1959, 0.303] | 0.1797 [0.131, 0.2355] |
| full | 0.2451 [0.1918, 0.3187] | 0.1841 [0.1341, 0.2378] |

## Selection (top 40 from the shortlist)

| Method | Recall@40 | Precision@40 | Facet coverage | Mean pairwise cosine |
|---|---|---|---|---|
| topk | 0.144 [0.1036, 0.1857] | 0.1357 | 1.0 | 0.8246 |
| mmr | 0.1405 [0.0976, 0.1869] | 0.1321 | 1.0 | 0.8138 |
| submodular | 0.1321 [0.075, 0.1857] | 0.1321 | 1.0 | 0.8206 |
| dpp | 0.1298 [0.0905, 0.1821] | 0.1214 | 1.0 | 0.7787 |

## Is the difference real?

| Metric | A | B | Mean A minus B | p-value |
|---|---|---|---|---|
| ndcg@100 | rrf_text | bm25f | 0.0744 | 0.0162 |
| ndcg@100 | rrf_text_ppr | rrf_text | 0.0512 | 0.0322 |
| ndcg@100 | full | rrf_text_ppr | 0.0043 | 0.4507 |
| recall@40 | submodular | topk | -0.0119 | 1.0 |
| recall@40 | dpp | topk | -0.0143 | 0.5675 |
| recall@40 | submodular | mmr | -0.0083 | 0.8743 |

## Judge

| Brain | Sensitivity | Specificity | Cohen's kappa |
|---|---|---|---|
| offline | 0.6917 | 0.4615 | 0.0627 |

## Per survey

| Survey | Pool | Answer key | Key papers in pool |
|---|---|---|---|
| gnn | 1778 | 84 | 83 |
| vision_transformers | 1959 | 213 | 122 |
| hallucination | 1477 | 57 | 24 |
| dense_retrieval | 1099 | 104 | 45 |
| contrastive_ssl | 1040 | 54 | 34 |
| federated | 1208 | 12 | 4 |
| scrnaseq | 1372 | 93 | 41 |

Caveats: with few surveys the intervals are wide. A survey does not cite every good paper, so recall is the
trustworthy number and precision is only a floor. OpenAlex's own order already mixes in citation counts.
