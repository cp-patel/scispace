# Selection eval

Surveys: 8. Values are means over surveys with 95% bootstrap intervals.

## Ranking

| Ranking | Recall@150 | nDCG@100 |
|---|---|---|
| openalex_order | 0.0591 [0.0296, 0.0896] | 0.0635 [0.0361, 0.0908] |
| bm25f | 0.0808 [0.0423, 0.1254] | 0.0668 [0.0319, 0.1073] |
| dense | 0.2046 [0.1811, 0.2315] | 0.1835 [0.1339, 0.2337] |
| rrf_text | 0.1875 [0.1698, 0.2115] | 0.1497 [0.1034, 0.2026] |
| rrf_text_ppr | 0.2414 [0.2002, 0.2963] | 0.1997 [0.1437, 0.2601] |
| full | 0.2453 [0.1976, 0.3112] | 0.2037 [0.1473, 0.2646] |

## Selection (top 40 from the shortlist)

| Method | Recall@40 | Precision@40 | Facet coverage | Mean pairwise cosine |
|---|---|---|---|---|
| topk | 0.1635 [0.1156, 0.2156] | 0.1562 | 1.0 | 0.8279 |
| mmr | 0.1542 [0.1083, 0.2] | 0.1469 | 1.0 | 0.814 |
| submodular | 0.15 [0.0906, 0.2062] | 0.15 | 1.0 | 0.8242 |
| dpp | 0.151 [0.0979, 0.2104] | 0.1438 | 1.0 | 0.7773 |

## Is the difference real?

| Metric | A | B | Mean A minus B | p-value |
|---|---|---|---|---|
| ndcg@100 | rrf_text | bm25f | 0.083 | 0.0081 |
| ndcg@100 | rrf_text_ppr | rrf_text | 0.05 | 0.0143 |
| ndcg@100 | full | rrf_text_ppr | 0.0039 | 0.4024 |
| recall@40 | submodular | topk | -0.0135 | 0.4958 |
| recall@40 | dpp | topk | -0.0125 | 0.5676 |
| recall@40 | submodular | mmr | -0.0042 | 0.9394 |

## Judge

| Brain | Sensitivity | Specificity | Cohen's kappa |
|---|---|---|---|
| offline | 0.7059 | 0.4642 | 0.0765 |

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
| materials_dl | 3000 | 300 | 79 |

Caveats: with few surveys the intervals are wide. A survey does not cite every good paper, so recall is the
trustworthy number and precision is only a floor. OpenAlex's own order already mixes in citation counts.
