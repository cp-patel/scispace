"""Selection eval: a published survey's reference list is the answer key.

For each survey: topic = its title; answer key = its references that have abstracts and were
published before it. The pipeline runs as if on the survey's publication date, with the survey
itself excluded, so the answer cannot leak in. Also measures the judge (sensitivity, specificity,
Cohen's kappa) and writes calibration.json, which production uses for the Bayes step.

Run: python -m evals.selection_eval [--surveys gnn,federated] [--k 40] [--shortlist 150]
A full first run needs a free OpenAlex key (or several days of the keyless allowance). Set
HTTP_CACHE_DIR so reruns are free.
"""

import argparse
import asyncio
import datetime as dt
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from research_assistant import funnel
from research_assistant.brain.base import Brain
from research_assistant.brain.offline import OfflineBrain
from research_assistant.config import AlgoParams, get_settings
from research_assistant.embed import Embedder, FastEmbedder
from research_assistant.ranking.bayes import calibration_table, judge_rates
from research_assistant.ranking.metrics import (
    bootstrap_ci,
    cohen_kappa,
    mean_pairwise_cosine,
    ndcg_at_k,
    paired_permutation_test,
    precision_at_k,
    recall_at_k,
)
from research_assistant.runtime import openalex_client
from research_assistant.sources.openalex import OpenAlex, short_id

SURVEYS = Path(__file__).with_name("surveys.json")
RESULTS = Path(__file__).with_name("results")
RANKINGS = {
    "openalex_order": None,
    "bm25f": {"use_dense": False, "use_ppr": False, "use_rocchio": False},
    "dense": {"use_bm25": False, "use_ppr": False, "use_rocchio": False},
    "rrf_text": {"use_ppr": False, "use_rocchio": False},
    "rrf_text_ppr": {"use_rocchio": False},
    "full": {},
}
SELECTORS = ("topk", "mmr", "submodular", "dpp")
COMPARISONS = [  # (metric group, metric, a, b): does a beat b?
    ("ranking", "ndcg@100", "rrf_text", "bm25f"),
    ("ranking", "ndcg@100", "rrf_text_ppr", "rrf_text"),
    ("ranking", "ndcg@100", "full", "rrf_text_ppr"),
    ("selection", "recall@40", "submodular", "topk"),
    ("selection", "recall@40", "dpp", "topk"),
    ("selection", "recall@40", "submodular", "mmr"),
]


@dataclass
class SurveyCase:
    key: str
    id: str
    title: str
    date: str
    answer: set[str]


async def load_case(oa: OpenAlex, entry: dict) -> SurveyCase:
    raw = await oa.get_raw(entry["id"])  # single-record lookups are free
    refs = [short_id(r) for r in raw.get("referenced_works") or []]
    date = raw["publication_date"]
    usable = await oa.by_ids(refs, published_before=date)
    return SurveyCase(entry["key"], entry["id"], raw["title"], date, {w.id for w in usable})


async def evaluate_case(case: SurveyCase, oa: OpenAlex, embedder: Embedder, brain: Brain, algo: AlgoParams,
                        shortlist_size: int = 150, k: int = 40) -> dict:
    plan = await brain.plan(case.title)
    pool, _ = await funnel.gather(plan, openalex=oa, embedder=embedder, algo=algo, published_before=case.date, exclude_ids=[case.id])
    if not pool:
        return {"survey": case.key, "pool": 0, "answer_key": len(case.answer), "in_pool": 0}
    vecs = embedder.embed_queries([f.question for f in plan.facets])
    row = {"survey": case.key, "pool": len(pool), "answer_key": len(case.answer),
           "in_pool": len({c.id for c in pool} & case.answer), "ranking": {}, "selection": {}}
    for name, flags in RANKINGS.items():
        if flags is None:
            scores = np.array([c.relevance_score if c.relevance_score is not None else -np.inf for c in pool])
        else:
            scores = funnel.compute_signals(pool, plan, vecs, algo, **flags).fused
        ranked = [pool[i].id for i in np.argsort(-scores, kind="stable")]
        row["ranking"][name] = {"recall@150": recall_at_k(ranked, case.answer, 150), "ndcg@100": ndcg_at_k(ranked, case.answer, 100)}
    sig = funnel.compute_signals(pool, plan, vecs, algo)
    short_idx = funnel.shortlist(sig, shortlist_size)
    short = [pool[i] for i in short_idx]
    rel = sig.facet_rel[short_idx]
    verdicts, post = await funnel.screen(brain, plan, short, rel, algo)
    row["judge"] = {
        "ranks": list(range(1, len(short) + 1)),
        "predicted": [bool(verdicts[c.id].include) if c.id in verdicts else False for c in short],
        "actual": [c.id in case.answer for c in short],
    }
    for method in SELECTORS:
        sel = funnel.select(method, short, post, rel, k, algo)
        ids = [short[i].id for i in sel.indices]
        covered = sum(1 for f in range(rel.shape[1]) if (rel[sel.indices, f] >= 0.5).sum() >= 3)
        row["selection"][method] = {
            "recall@40": recall_at_k(ids, case.answer, k),
            "precision@40": precision_at_k(ids, case.answer, k),
            "facet_coverage": covered / rel.shape[1],
            "diversity": mean_pairwise_cosine(np.stack([short[i].embedding for i in sel.indices])),
        }
    return row


def summarize(rows: list[dict]) -> dict:
    rows = [r for r in rows if r.get("pool")]
    summary: dict = {"surveys": len(rows), "ranking": {}, "selection": {}, "comparisons": []}
    for group in ("ranking", "selection"):
        for name in rows[0][group]:
            summary[group][name] = {}
            for metric in rows[0][group][name]:
                values = np.array([r[group][name][metric] for r in rows])
                low, high = bootstrap_ci(values) if len(values) > 1 else (values[0], values[0])
                summary[group][name][metric] = {"mean": round(float(values.mean()), 4), "ci95": [round(low, 4), round(high, 4)]}
    for group, metric, a, b in COMPARISONS:
        va = np.array([r[group][a][metric] for r in rows])
        vb = np.array([r[group][b][metric] for r in rows])
        summary["comparisons"].append({"metric": metric, "a": a, "b": b, "mean_difference": round(float((va - vb).mean()), 4),
                                       "p_value": round(paired_permutation_test(va, vb), 4) if len(va) > 1 else None})
    return summary


def calibration(rows: list[dict], brain_name: str, previous: dict | None) -> dict:
    ranks = [r for row in rows if row.get("judge") for r in row["judge"]["ranks"]]
    predicted = [p for row in rows if row.get("judge") for p in row["judge"]["predicted"]]
    actual = [a for row in rows if row.get("judge") for a in row["judge"]["actual"]]
    rates = judge_rates(predicted, actual)
    out = {"prior_table": calibration_table(ranks, actual), "judge_rates": dict((previous or {}).get("judge_rates", {}))}
    out["judge_rates"][brain_name] = {"sensitivity": round(rates.sensitivity, 4), "specificity": round(rates.specificity, 4),
                                      "kappa": round(cohen_kappa(predicted, actual), 4)}
    return out


def markdown(summary: dict, rows: list[dict], cal: dict) -> str:
    lines = ["# Selection eval", "", f"Surveys: {summary['surveys']}. Values are means over surveys with 95% bootstrap intervals.", ""]
    lines += ["## Ranking", "", "| Ranking | Recall@150 | nDCG@100 |", "|---|---|---|"]
    for name, m in summary["ranking"].items():
        lines.append(f"| {name} | {m['recall@150']['mean']} {m['recall@150']['ci95']} | {m['ndcg@100']['mean']} {m['ndcg@100']['ci95']} |")
    lines += ["", "## Selection (top 40 from the shortlist)", "",
              "| Method | Recall@40 | Precision@40 | Facet coverage | Mean pairwise cosine |", "|---|---|---|---|---|"]
    for name, m in summary["selection"].items():
        lines.append(f"| {name} | {m['recall@40']['mean']} {m['recall@40']['ci95']} | {m['precision@40']['mean']} | "
                     f"{m['facet_coverage']['mean']} | {m['diversity']['mean']} |")
    lines += ["", "## Is the difference real?", "", "| Metric | A | B | Mean A minus B | p-value |", "|---|---|---|---|---|"]
    lines += [f"| {c['metric']} | {c['a']} | {c['b']} | {c['mean_difference']} | {c['p_value']} |" for c in summary["comparisons"]]
    lines += ["", "## Judge", "", "| Brain | Sensitivity | Specificity | Cohen's kappa |", "|---|---|---|---|"]
    lines += [f"| {b} | {r['sensitivity']} | {r['specificity']} | {r['kappa']} |" for b, r in cal["judge_rates"].items()]
    lines += ["", "## Per survey", "", "| Survey | Pool | Answer key | Key papers in pool |", "|---|---|---|---|"]
    lines += [f"| {r['survey']} | {r['pool']} | {r['answer_key']} | {r['in_pool']} |" for r in rows]
    lines += ["", "Caveats: with few surveys the intervals are wide. A survey does not cite every good paper, so recall is the",
              "trustworthy number and precision is only a floor. OpenAlex's own order already mixes in citation counts.", ""]
    return "\n".join(lines)


async def run(keys: list[str] | None, k: int, shortlist_size: int) -> None:
    settings = get_settings()
    entries = [e for e in json.loads(SURVEYS.read_text()) if keys is None or e["key"] in keys]
    oa = openalex_client(settings)
    embedder = FastEmbedder(settings.embed_model, settings.embed_cache_dir)
    brain = OfflineBrain(settings.algo.max_cards)
    rows = []
    try:
        for entry in entries:
            case = await load_case(oa, entry)
            print(f"{case.key}: answer key has {len(case.answer)} papers")
            rows.append(await evaluate_case(case, oa, embedder, brain, settings.algo, shortlist_size, k))
    finally:
        await oa.http.aclose()
    RESULTS.mkdir(exist_ok=True)
    cal_path = RESULTS / "calibration.json"
    previous = json.loads(cal_path.read_text()) if cal_path.exists() else None
    cal = calibration(rows, brain.name, previous)
    summary = summarize(rows)
    stamp = dt.date.today().isoformat()
    (RESULTS / f"selection-{stamp}.json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=2, default=float))
    (RESULTS / f"selection-{stamp}.md").write_text(markdown(summary, rows, cal))
    cal_path.write_text(json.dumps(cal, indent=2))
    print(f"OpenAlex cost this run: ${oa.http.cost_usd:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--surveys", help="comma-separated keys from surveys.json (default: all)")
    parser.add_argument("--k", type=int, default=40)
    parser.add_argument("--shortlist", type=int, default=150)
    args = parser.parse_args()
    asyncio.run(run(args.surveys.split(",") if args.surveys else None, args.k, args.shortlist))


if __name__ == "__main__":
    main()
