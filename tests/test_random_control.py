"""Known-answer and harness checks for continuous random acquisition."""


import numpy as np
import pytest

from screenprior import random_control as CONTROL


def test_hand_computable_stream_is_consumed_once(monkeypatch):
    """A reset after the opening cannot pass this three-batch known-answer path."""
    class KnownStream:
        def __init__(self):
            self.calls = 0

        def choice(self, pool, size, replace):
            assert size == 2 and replace is False
            expected_pool = [np.arange(6), np.array([0, 2, 3, 5]), np.array([2, 3])][self.calls]
            observed_pool = np.arange(pool) if np.isscalar(pool) else pool
            np.testing.assert_array_equal(observed_pool, expected_pool)
            answer = [[1, 4], [0, 5], [3, 2]][self.calls]
            self.calls += 1
            return np.array(answer)

    stream = KnownStream()
    initializations = []

    def initialize(seed):
        initializations.append(seed)
        return stream

    monkeypatch.setattr(CONTROL.np.random, "default_rng", initialize)
    actual = CONTROL.continuous_random_indices(6, 11, batch=2, cycles=3)
    np.testing.assert_array_equal(actual, [1, 4, 0, 5, 3, 2])
    assert initializations == [11]
    assert stream.calls == 3


@pytest.mark.parametrize("seed", range(10))
def test_wrapper_matches_a_single_continuous_generator(seed):
    """The registry wrapper must preserve every draw, including the shared first 64."""
    n, batch = 5000, 64
    reference_rng = np.random.default_rng(seed)
    expected = reference_rng.choice(n, size=batch, replace=False)
    original_first = expected.copy()
    policy = CONTROL.corrected_random_policy(seed, n, batch)
    actual = original_first.copy()
    for _ in range(9):
        residual = np.setdiff1d(np.arange(n), actual)
        picked = policy(np.random.default_rng(999), residual, batch, chosen=actual)
        actual = np.concatenate([actual, picked])
        expected_pool = np.setdiff1d(np.arange(n), expected)
        expected = np.concatenate([
            expected, reference_rng.choice(expected_pool, size=batch, replace=False)])
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(actual, CONTROL.continuous_random_indices(n, seed))
    np.testing.assert_array_equal(actual[:64], original_first)
    assert len(np.unique(actual)) == 640
    assert actual.min() >= 0 and actual.max() < n


def test_labels_and_supplied_slot_rng_do_not_affect_acquisitions():
    """A random control may not read the answer key or a restarted slot RNG."""
    n, seed, batch = 5000, 0, 64
    first = np.random.default_rng(seed).choice(n, size=batch, replace=False)
    residual = np.setdiff1d(np.arange(n), first)
    left = CONTROL.corrected_random_policy(seed, n, batch)
    right = CONTROL.corrected_random_policy(seed, n, batch)
    a = left(np.random.default_rng(0), residual, batch, chosen=first, observed=np.zeros(n))
    b = right(np.random.default_rng(900), residual, batch, chosen=first, observed=np.ones(n))
    np.testing.assert_array_equal(a, b)


def test_mismatched_harness_opening_is_rejected():
    """A closure for the wrong target/seed cannot silently continue a different opening."""
    policy = CONTROL.corrected_random_policy(0, 5000)
    wrong = np.random.default_rng(1).choice(5000, size=64, replace=False)
    with pytest.raises(ValueError, match="history differs"):
        policy(None, np.setdiff1d(np.arange(5000), wrong), 64, chosen=wrong)


def test_harness_registry_integration_keeps_recorded_picks_label_free():
    """The scoped wrapper works through the released harness with the shared opening."""
    from screenprior.corpus import Corpus
    from screenprior.harness import RunConfig, run_screen

    n, seed = 1000, 3
    labels = np.arange(n) % 11 == 0
    hit = np.column_stack([labels, ~labels])
    corpus = Corpus(
        genes=np.array([f"G{i:04d}" for i in range(n)]), screen_id=np.array(["target", "donor"]),
        pmid=np.array(["P1", "P2"]), phenotype=np.array(["test", "test"]),
        cell_line=np.array(["line1", "line2"]),
        hit=hit, tested=np.ones_like(hit))
    config = RunConfig(seeds=(seed,), pool="full", min_pool=1, min_hits=1, record_picks=True)
    policy = CONTROL.corrected_random_policy(seed, n)
    rows, traces = run_screen(
        corpus, 0, np.zeros((n, 1)), np.zeros(n), policies=("random",), config=config,
        registry={"random": policy})
    expected = CONTROL.continuous_random_indices(n, seed)
    np.testing.assert_array_equal(traces[0]["picks"], expected)
    assert rows[0]["acquired"] == 640
    assert rows[0]["recall"] == labels[expected].sum() / labels.sum()
