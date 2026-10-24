"""Endpoints, and the denominators that distinguish them.

Four numbers get called "hit ratio" in this literature and they are not the same quantity. Getting
them confused is how a table ends up comparing a paper's precision against another's recall, so
each one is a separate function here with its denominator in the docstring, and `chance` exists so
that every reported value can be divided by what a coin would score.

  recall_at        acquired hits / all hits in the pool            GeneDisco's HitRatio
  precision_at     acquired hits / number of acquisitions          OAE's "ratio of arm pulls"
  lift             either of the above / the base rate             puts a value on a chance scale
  auprc_lift       average precision / base rate                   no budget-dependent ceiling

The truncation in the first two is not cosmetic. `precision_at(k)` cannot exceed
min(k, n_hits) / k, so a screen with fewer hits than the budget caps below 1.0 and a perfect
ranker's score becomes a fact about the screen rather than about the ranker. Measured on this
project's own data, that effect moved a "perfect" arm from 20.5 to 11.6 between two label regimes
and briefly read as a method result. `auprc_lift` has no such ceiling. It is NOT pool-size invariant --
measured, halving the candidate set moves it about 10% -- so two conditions that score different
candidate sets are still not comparable; the fix for that is to score one fixed held-out set in
both.
"""

from __future__ import annotations

import numpy as np
from scipy import stats
from sklearn.metrics import average_precision_score

__all__ = ["auprc_lift", "chance", "lift", "perfect_precision", "precision_at", "recall_at",
           "signed_rank_p"]


def _check(scores: np.ndarray, labels: np.ndarray) -> None:
    if len(scores) != len(labels):
        raise ValueError(f"scores and labels differ in length: {len(scores)} vs {len(labels)}")
    if not np.isfinite(scores).all():
        raise ValueError("scores contain non-finite values; a ranking cannot be formed from them")
    if labels.sum() == 0:
        raise ValueError("no positives among the labels; every ratio would divide by zero")


def chance(budget: int, n_candidates: int) -> float:
    """What uniform random acquisition recovers, as a fraction of the pool's hits."""
    if n_candidates <= 0:
        raise ValueError("n_candidates must be positive")
    return budget / n_candidates


def _top(scores: np.ndarray, budget: int) -> np.ndarray:
    # `kind="stable"` so that ties resolve by position rather than by whatever the sort does that
    # day; two arms fed the same scores must select the same genes.
    return np.argsort(-scores, kind="stable")[:budget]


def recall_at(scores: np.ndarray, labels: np.ndarray, budget: int) -> float:
    """Acquired hits divided by ALL hits in the pool. GeneDisco's HitRatio."""
    _check(scores, labels)
    return float(labels[_top(scores, budget)].sum()) / float(labels.sum())


def precision_at(scores: np.ndarray, labels: np.ndarray, budget: int) -> float:
    """Acquired hits divided by the NUMBER OF ACQUISITIONS. OAE's "ratio of arm pulls"."""
    _check(scores, labels)
    return float(labels[_top(scores, budget)].mean())


def perfect_precision(labels: np.ndarray, budget: int) -> float:
    """The ceiling `precision_at` can reach on this pool, which is not always 1.0."""
    return min(budget, int(labels.sum())) / float(budget)


def lift(value: float, labels: np.ndarray) -> float:
    """A precision divided by the base rate: 1.0 means indistinguishable from random."""
    base = float(np.asarray(labels).mean())
    if base <= 0:
        raise ValueError("base rate is zero")
    return value / base


def auprc_lift(scores: np.ndarray, labels: np.ndarray) -> float:
    """Average precision over the base rate. No budget-dependent ceiling.

    Prefer this over a top-k metric whenever the BUDGET varies, because a top-k metric silently
    changes its own ceiling and the difference reads as an effect. It does not licence comparing
    two conditions that score different candidate sets -- for that, score one fixed held-out set
    in both.
    """
    _check(scores, labels)
    return float(average_precision_score(labels, scores)) / float(np.asarray(labels).mean())


def signed_rank_p(d, zero_method: str = "wilcox") -> float:
    """Two-sided Wilcoxon signed-rank P, with the exact-or-normal choice made here, not by scipy.

    scipy's `method="auto"` has not meant the same thing in every release. 1.14 takes the exact
    null distribution for n <= 50 even when the |d| contain ties, and that distribution assumes
    there are none; 1.18 takes the normal approximation with the tie correction instead.
    Measured on the
    GeneDisco comparison at 64 genes per cycle (n = 50, seven tied |d|): 0.0103 under 1.14's auto
    and 0.0110 under the approximation, where the manuscript then said 0.011. Exact only where it
    is valid: n <= 50, no zero differences and no tied magnitudes.
    """
    d = np.asarray(d, dtype=float)
    magnitudes = np.abs(d[d != 0])
    exact = (len(d) <= 50 and len(magnitudes) == len(d)
             and len(np.unique(magnitudes)) == len(magnitudes))
    method = "exact" if exact else "approx"
    return float(stats.wilcoxon(d, zero_method=zero_method, method=method).pvalue)
