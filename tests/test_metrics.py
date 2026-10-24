"""The four things called "hit ratio" are four different numbers. These tests pin which is which."""

from __future__ import annotations

import numpy as np
import pytest

from screenprior.metrics import (
    auprc_lift,
    chance,
    lift,
    perfect_precision,
    precision_at,
    recall_at,
    signed_rank_p,
)


def test_recall_and_precision_differ_by_their_denominator():
    # Ten candidates, four hits, budget three. A ranker that puts two hits in the top three:
    scores = np.array([9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0])
    labels = np.array([1, 0, 1, 0, 1, 0, 0, 1, 0, 0])
    assert precision_at(scores, labels, 3) == pytest.approx(2 / 3)   # denominator = acquisitions
    assert recall_at(scores, labels, 3) == pytest.approx(2 / 4)      # denominator = all hits


def test_precision_ceiling_is_a_fact_about_the_screen_not_the_ranker():
    """A pool with fewer hits than the budget caps precision below 1.0 for ANY ranker.

    This is the trap that once made a `perfect` arm read 20.5 in one label regime and 11.6 in
    another and look like a method effect. `perfect_precision` states the ceiling explicitly.
    """
    labels = np.array([1, 1, 0, 0, 0, 0, 0, 0])
    oracle = labels.astype(float)                     # a ranker that cannot be improved on
    assert precision_at(oracle, labels, 4) == pytest.approx(0.5)
    assert perfect_precision(labels, 4) == pytest.approx(0.5)
    assert recall_at(oracle, labels, 4) == pytest.approx(1.0)      # recall has no such ceiling


def test_auprc_lift_is_one_for_an_uninformative_score():
    rng = np.random.default_rng(0)
    labels = (rng.random(20_000) < 0.08).astype(int)
    noise = rng.random(20_000)
    assert auprc_lift(noise, labels) == pytest.approx(1.0, abs=0.05)


def test_auprc_lift_has_no_budget_dependent_ceiling_but_is_not_pool_invariant():
    """The property `auprc_lift` actually has, and the one it does NOT have.

    An earlier version of this test asserted that the value is invariant to pool size. It is not:
    measured here, halving the pool moved it by about 10%. What `auprc_lift` removes is the
    BUDGET-DEPENDENT CEILING that `precision_at` carries, which is the artifact that once read as a
    method effect. Two conditions that score different candidate sets still are not comparable —
    the fix for that is to score one fixed held-out set in both, not to change the metric.
    """
    rng = np.random.default_rng(1)
    labels = (rng.random(40_000) < 0.05).astype(int)
    base = labels.mean()

    # A perfect ranking has average precision 1.0 for ANY budget, so its lift is exactly 1/base.
    assert auprc_lift(labels.astype(float) + rng.random(40_000) * 1e-9, labels) == pytest.approx(
        1.0 / base, rel=1e-6)

    # `precision_at`'s ceiling, by contrast, is a function of the budget alone.
    n_hits = int(labels.sum())
    assert perfect_precision(labels, n_hits // 2) == pytest.approx(1.0)
    assert perfect_precision(labels, n_hits * 2) == pytest.approx(0.5)


def test_lift_of_random_is_one_and_chance_is_the_budget_share():
    labels = np.array([1, 0, 0, 0, 1, 0, 0, 0, 0, 0])
    assert lift(0.2, labels) == pytest.approx(1.0)
    assert chance(640, 16_000) == pytest.approx(0.04)


def test_ties_resolve_by_position_so_two_arms_cannot_disagree_on_identical_scores():
    scores = np.zeros(10)
    labels = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1])
    assert precision_at(scores, labels, 5) == pytest.approx(0.0)
    assert precision_at(scores.copy(), labels, 5) == precision_at(scores, labels, 5)


@pytest.mark.parametrize("bad, message", [
    (np.array([np.nan, 1.0, 2.0]), "non-finite"),
    (np.array([1.0, 2.0]), "differ in length"),
])
def test_malformed_input_raises_rather_than_scoring(bad, message):
    labels = np.array([1, 0, 1])
    with pytest.raises(ValueError, match=message):
        recall_at(bad, labels, 2)


def test_no_positives_raises_rather_than_returning_zero():
    with pytest.raises(ValueError, match="no positives"):
        recall_at(np.array([1.0, 2.0, 3.0]), np.zeros(3, dtype=int), 2)


def test_signed_rank_p_does_not_take_the_exact_distribution_when_magnitudes_tie():
    """The exact null assumes untied |d|; scipy 1.14's auto used it anyway and moved a reported P.

    Ten equal positive differences: every rank is 5.5, so T- = 0. With the tie correction the
    variance is 10*11*21/24 - (10**3 - 10)/48 = 96.25 - 20.625 = 75.625, z = -27.5 / 8.6963
    = -3.1623, and the two-sided P is 2 * Phi(-3.1623) = 0.001565. The exact table, which does not
    know the ranks are tied, gives 2 / 2**10 = 0.00195.
    """
    assert signed_rank_p([1.0] * 10) == pytest.approx(0.001565, abs=2e-5)


def test_signed_rank_p_is_exact_when_the_exact_distribution_is_valid():
    """Five distinct positive differences: T- = 0 is one sign pattern of 32, twice for two sides."""
    assert signed_rank_p([1.0, 2.0, 3.0, 4.0, 5.0]) == pytest.approx(2 / 32)
