"""Continuous-RNG random control for sequential acquisition.

The shared opening remains default_rng(seed).choice(n, batch, replace=False).
Only this random control continues that generator after its opening draw.
The existing harness streams and nine-arm ordering remain unchanged.
"""

from __future__ import annotations

import numpy as np


def _dimensions(n: int, batch: int, cycles: int = 1) -> None:
    if any(int(value) != value for value in (n, batch, cycles)):
        raise ValueError("n, batch and cycles must be integers")
    if batch < 1 or cycles < 1 or n < batch * cycles:
        raise ValueError("need a positive batch/cycle count and enough distinct genes")


def continuous_random_indices(n: int, seed: int, batch: int = 64,
                              cycles: int = 10) -> np.ndarray:
    """Uniform sampling without replacement using one continuously advanced RNG."""
    _dimensions(n, batch, cycles)
    rng = np.random.default_rng(seed)
    chosen = rng.choice(n, size=batch, replace=False)
    for _ in range(1, cycles):
        pool = np.setdiff1d(np.arange(n), chosen)
        chosen = np.concatenate([chosen, rng.choice(pool, size=batch, replace=False)])
    return chosen


def corrected_random_policy(seed: int, n: int, batch: int = 64):
    """Make a fresh harness registry policy for exactly one target and one seed.

    Call run_screen with config.seeds=(seed,), and replace only registry['random'].
    Retain the original full policy order for a multi-arm run. The harness's
    slot RNG is intentionally unused: this policy consumes the original shared
    opening in its own generator, then continuously advances that generator.
    The closure refuses inconsistent pools/history rather than restarting.
    """
    _dimensions(n, batch)
    configured_batch = batch
    rng = np.random.default_rng(seed)
    history = rng.choice(n, size=batch, replace=False)

    def policy(_slot_rng, pool, batch, *, chosen=None, **_):
        nonlocal history
        if batch != configured_batch:
            raise ValueError("random control batch changed after its opening")
        if chosen is not None and not np.array_equal(chosen, history):
            raise ValueError("random control history differs from the harness opening/continuation")
        expected_pool = np.setdiff1d(np.arange(n), history)
        if not np.array_equal(pool, expected_pool):
            raise ValueError("random control requires the sorted residual pool for its own history")
        if len(pool) < batch:
            raise ValueError("random control has fewer remaining genes than its next batch")
        picked = rng.choice(pool, size=batch, replace=False)
        history = np.concatenate([history, picked])
        return picked

    return policy
