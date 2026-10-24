"""Hand-computed checks on the DiscoBAX comparator, written against the paper's equations.

Deliberately NOT a comparison against a transcription of the authors' code. The upstream licence is
self-contradictory -- GPL-3.0 in the root LICENSE, Apache-2.0 in the source headers -- so the
verification here is arithmetic done by hand from equations (5), (6) and Algorithm 1, which is both
licence-clean and a stronger check: a transcription can only prove two implementations agree, while
these prove the implementation computes the quantity the paper defines.
"""

from __future__ import annotations

import numpy as np
import pytest

from screenprior.comparators import (
    DISCOBAX_CONFIG,
    GaussianFieldNoise,
    expected_information_gain,
    predictive_entropy,
    rbf_kernel,
    subset_select,
)


def test_subset_select_prefers_complementary_candidates_over_a_strong_pair():
    """The whole point of equation (5): two candidates that fire together are worth less than two
    that cover different draws.

    Four draws, three candidates. A is best on draws 0 and 1, B is identical to A, C is best on
    draws 2 and 3.
        mean(A) = mean(B) = (9 + 9 + 0 + 0) / 4 = 4.5   -> A is picked first
        adding B: mean max(A, B) = 4.5                  -> no gain at all
        adding C: mean max(A, C) = (9 + 9 + 8 + 8) / 4 = 8.5
    So the greedy step must take C second even though B ties A on its own.
    """
    values = np.array([[9.0, 9.0, 0.0],
                       [9.0, 9.0, 0.0],
                       [0.0, 0.0, 8.0],
                       [0.0, 0.0, 8.0]])
    assert subset_select(values, 2) == [0, 2]


def test_subset_select_first_pick_is_the_largest_mean():
    values = np.array([[1.0, 5.0, 3.0], [1.0, 5.0, 3.0]])
    assert subset_select(values, 1) == [1]


def test_subset_select_gain_arithmetic_matches_a_hand_computation():
    values = np.array([[4.0, 3.0, 0.0], [0.0, 3.0, 6.0]])
    # means: 2.0, 3.0, 3.0 -> ties on 1 and 2; argmax takes the first, so the seed is candidate 1.
    # then max(v1, v0) = [4, 3] -> 3.5 ; max(v1, v2) = [3, 6] -> 4.5 -> candidate 2 wins.
    assert subset_select(values, 2) == [1, 2]


def test_subset_select_rejects_a_size_it_cannot_satisfy():
    with pytest.raises(ValueError, match="outside 1.."):
        subset_select(np.zeros((3, 4)), 5)


def test_predictive_entropy_is_larger_for_more_scattered_samples():
    """Equation (6) is an entropy: spread-out predictions must score higher than agreeing ones."""
    tight = np.tile(np.array([0.0, 0.01, -0.01, 0.0]), (3, 1))
    loose = np.tile(np.array([-5.0, 5.0, -2.0, 2.0]), (3, 1))
    assert (predictive_entropy(loose) > predictive_entropy(tight)).all()


def test_predictive_entropy_matches_a_hand_computed_two_sample_case():
    """M = 2 samples at 0 and d. The kernel estimator's log density at each sample is
    log(1 + exp(-d^2/2)), identical for both, so

        H = -( log(1 + exp(-d^2/2)) - log 2 - 0.5 log(2 pi) + 0 )
    """
    for d in (0.0, 1.0, 3.0):
        got = predictive_entropy(np.array([[0.0, d]]))[0]
        expected = -(np.log(1 + np.exp(-d ** 2 / 2)) - np.log(2) - 0.5 * np.log(2 * np.pi))
        assert got == pytest.approx(expected, rel=1e-10)


def test_information_gain_is_zero_when_the_sets_change_nothing():
    prior = np.array([[0.0, 1.0, 2.0], [1.0, 1.0, 1.0]])
    posterior = np.stack([prior, prior])
    assert np.allclose(expected_information_gain(prior, posterior), 0.0)


def test_information_gain_is_positive_when_the_sets_sharpen_the_prediction():
    prior = np.array([[-4.0, 0.0, 4.0]])
    posterior = np.stack([np.array([[0.0, 0.0, 0.01]])] * 3)
    assert expected_information_gain(prior, posterior)[0] > 0


def test_information_gain_rejects_a_wrongly_shaped_posterior():
    with pytest.raises(ValueError, match="sets, candidates, draws"):
        expected_information_gain(np.zeros((2, 3)), np.zeros((2, 3)))


def test_rbf_kernel_is_one_on_the_diagonal_and_decays_with_distance():
    X = np.array([[0.0], [1.0], [3.0]])
    K = rbf_kernel(X, X)
    assert np.allclose(np.diag(K), 1.0)
    assert K[0, 1] == pytest.approx(np.exp(-0.5))
    assert K[0, 2] == pytest.approx(np.exp(-4.5))
    assert K[0, 2] < K[0, 1]


def test_the_noise_field_reproduces_the_kernel_it_was_built_from():
    """The sampler is exact, so its Gram matrix IS the kernel -- no Monte-Carlo error involved."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(120, 4))
    noise = GaussianFieldNoise(X)
    root = noise.root.astype(np.float64)
    assert np.abs(root @ root.T - rbf_kernel(X, X)).max() < 1e-5


def test_the_noise_field_is_spatially_correlated_not_independent():
    """If the draws were independent per gene the diversity term would do nothing, which is the
    failure mode that standardising the features silently produced during development."""
    X = np.linspace(0, 1, 60)[:, None] * 0.3          # points close together at lengthscale 1
    noise = GaussianFieldNoise(X)
    draws = np.array([noise.draw(np.random.default_rng(s)) for s in range(400)])
    neighbours = np.corrcoef(draws[:, 0], draws[:, 1])[0, 1]
    assert neighbours > 0.9


def test_the_configuration_records_what_the_paper_does_not_state():
    assert DISCOBAX_CONFIG["subset_size"] == 20
    assert set(DISCOBAX_CONFIG) == {"subset_size", "num_samples_eig", "num_samples_entropy",
                                    "noise_lengthscale", "noise_outputscale", "noise_draws"}
