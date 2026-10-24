"""Paired inference and row validation for the saved AssayFormer comparison."""
from __future__ import annotations

from itertools import product

import numpy as np
import pandas as pd
from frozen_metrics import signed_rank_p as _signed_rank_p
from row_validation import BUDGET, F2_SIZE, exact_cells, numeric

SEEDS = 10
POOLS = ("matched", "native")


def signed_rank_p(values) -> float:
    """An all-zero paired contrast has null p=1; otherwise retain the frozen routine."""
    differences = np.asarray(values, dtype=float)
    if len(differences) and np.all(differences == 0):
        return 1.0
    return _signed_rank_p(differences)


def bootstrap(values: np.ndarray, groups: np.ndarray | None = None, n: int = 10_000,
              seed: int = 0) -> list[float]:
    """Percentile interval of the mean, resampling screens, or whole groups when given."""
    rng = np.random.default_rng(seed)
    if groups is None:
        draws = rng.choice(values, size=(n, len(values)), replace=True).mean(axis=1)
    else:
        labels = np.unique(groups)
        members = [values[groups == g] for g in labels]
        draws = np.empty(n)
        for i in range(n):
            pick = rng.integers(0, len(labels), size=len(labels))
            draws[i] = np.concatenate([members[j] for j in pick]).mean()
    return [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))]


def validate_assayformer_rows(af: pd.DataFrame, route: pd.DataFrame,
                             names: list[str]) -> pd.DataFrame:
    """Refuse corrupt/incomplete cells before aggregation or paired inference."""
    frame = af.copy()
    if {"screen", "pool"} - set(frame.columns):
        raise ValueError("AssayFormer: missing screen or pool column")
    counts = ("seed", "found", "total_hits", "n_genes", "n_hits", "library_size", "n_picks",
              "n_in_library", "n_out_universe", "n_unfilled")
    numeric(frame, counts + ("ef",), "AssayFormer")
    if any((frame[c] != np.floor(frame[c])).any() for c in counts):
        raise ValueError("AssayFormer: seeds and counts must be integers")
    exact_cells(frame, ("screen", "seed", "pool"),
                set(product(names, range(SEEDS), POOLS)), "AssayFormer")
    if ((frame[list(counts)] < 0).any().any() or (frame.ef < 0).any() or
            (frame.n_picks != BUDGET).any() or (frame.n_genes < BUDGET).any() or
            (frame.n_hits > frame.n_genes).any() or (frame.n_hits > frame.total_hits).any() or
            (frame.total_hits <= 0).any() or (frame.total_hits > frame.library_size).any() or
            (frame.n_in_library > BUDGET).any() or (frame.found > frame.n_in_library).any() or
            (frame.found > frame.n_hits).any() or (frame.n_out_universe != 0).any() or
            (frame.n_unfilled != 0).any()):
        raise ValueError("AssayFormer: invalid budget, pool counts, hit counts or EF")
    for column in ("n_genes", "n_hits", "total_hits", "library_size"):
        if (frame.groupby(["screen", "pool"])[column].nunique() != 1).any():
            raise ValueError(f"AssayFormer: {column} changes across a screen/pool's seeds")
    reference = route.groupby("screen").first().reindex(frame.screen)
    for column in ("total_hits", "library_size"):
        if not np.array_equal(frame[column].to_numpy(), reference[column].to_numpy()):
            raise ValueError(f"AssayFormer: full-library {column} differs from frozen A.1")
    matched = frame.pool == "matched"
    for column in ("n_genes", "n_hits"):
        if not np.array_equal(frame.loc[matched, column].to_numpy(),
                              reference.loc[matched.to_numpy(), column].to_numpy()):
            raise ValueError(f"AssayFormer: matched {column} differs from frozen A.1")
    if ((frame.loc[~matched, "n_genes"] != F2_SIZE).any() or
            (frame.loc[matched, "n_in_library"] != BUDGET).any()):
        raise ValueError("AssayFormer: native f2 size or matched in-library budget is wrong")
    pool_hits = frame.groupby(["screen", "pool"]).n_hits.first().unstack("pool")
    if (pool_hits.matched > pool_hits.native).any():
        raise ValueError("AssayFormer: matched pool has more hits than its native superset")
    effective = frame.n_in_library + frame.n_out_universe + frame.n_unfilled
    denominator = effective * frame.total_hits / frame.library_size
    expected_ef = np.divide(frame.found, denominator, out=np.zeros(len(frame)),
                            where=denominator.to_numpy() > 0)
    if not np.allclose(frame.ef, expected_ef, rtol=1e-10, atol=1e-10):
        raise ValueError("AssayFormer: EF disagrees with the frozen full-library denominator")
    if "frac_hits" in frame:
        numeric(frame, ("frac_hits",), "AssayFormer")
        if not np.allclose(frame.frac_hits, frame.found / frame.total_hits,
                           rtol=1e-10, atol=1e-10):
            raise ValueError("AssayFormer: fraction of hits disagrees with found/total_hits")
    return frame
