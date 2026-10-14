"""Turn an LLM judge's include/exclude verdict into a probability with Bayes' rule.

The judge is treated as a noisy test with measured error rates, like a medical test:
sensitivity = P(says include | relevant), specificity = P(says exclude | not relevant).
"""

from dataclasses import dataclass

EPS = 1e-6
DEFAULT_BUCKETS: tuple[tuple[int, int | None], ...] = ((1, 10), (11, 25), (26, 50), (51, 100), (101, None))


@dataclass(frozen=True)
class JudgeRates:
    sensitivity: float
    specificity: float


def _clip(p: float) -> float:
    return min(max(p, EPS), 1.0 - EPS)


def linear_prior(rank: int, m: int, top: float = 0.6, bottom: float = 0.1) -> float:
    """Fallback prior before calibration exists: top at rank 1, falling linearly to bottom at rank m."""
    if m <= 1:
        return top
    fraction = (min(max(rank, 1), m) - 1) / (m - 1)
    return top + (bottom - top) * fraction


def prior_from_table(rank: int, table: list[tuple[int, int | None, float]]) -> float:
    """Look up the observed fraction of relevant papers for this rank's bucket."""
    for low, high, p in table:
        if rank >= low and (high is None or rank <= high):
            return p
    raise ValueError(f"rank {rank} is not covered by the calibration table")


def likelihood_ratio(include: bool, rates: JudgeRates) -> float:
    """LR+ = Se / (1 - Sp) for include; LR- = (1 - Se) / Sp for exclude."""
    se, sp = _clip(rates.sensitivity), _clip(rates.specificity)
    return se / (1.0 - sp) if include else (1.0 - se) / sp


def posterior(prior: float, include: bool, rates: JudgeRates) -> float:
    """posterior odds = prior odds * likelihood ratio, then back to a probability."""
    p = _clip(prior)
    odds = p / (1.0 - p) * likelihood_ratio(include, rates)
    return odds / (1.0 + odds)


def calibration_table(
    ranks: list[int],
    relevant: list[bool],
    buckets: tuple[tuple[int, int | None], ...] = DEFAULT_BUCKETS,
) -> list[tuple[int, int | None, float]]:
    """Fraction of relevant papers in each rank bucket, with add-one smoothing so no bucket is exactly 0 or 1."""
    table = []
    for low, high in buckets:
        hits = [rel for r, rel in zip(ranks, relevant) if r >= low and (high is None or r <= high)]
        table.append((low, high, (sum(hits) + 1) / (len(hits) + 2)))
    return table


def judge_rates(predicted: list[bool], actual: list[bool]) -> JudgeRates:
    """Measure sensitivity and specificity against labels, with add-one smoothing."""
    tp = sum(1 for p, a in zip(predicted, actual) if p and a)
    fn = sum(1 for p, a in zip(predicted, actual) if not p and a)
    tn = sum(1 for p, a in zip(predicted, actual) if not p and not a)
    fp = sum(1 for p, a in zip(predicted, actual) if p and not a)
    return JudgeRates(sensitivity=(tp + 1) / (tp + fn + 2), specificity=(tn + 1) / (tn + fp + 2))
