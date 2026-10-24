"""Validate the 20-screen saved-result grid and its integer metric denominators."""
from __future__ import annotations

from itertools import product

import numpy as np
import pandas as pd

SEEDS = tuple(range(10))
BATCH, STEPS, BUDGET = 100, 10, 1000
F2_SIZE = 21_147


def numeric(frame: pd.DataFrame, columns: tuple[str, ...], label: str) -> None:
    """Reject missing/nonfinite quantities before any grouping can conceal them."""
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(f"{label}: missing columns {sorted(missing)}")
    for column in columns:
        values = pd.to_numeric(frame[column], errors="coerce")
        if not np.isfinite(values.to_numpy(dtype=float)).all():
            raise ValueError(f"{label}: {column} contains a missing or nonfinite value")
        frame[column] = values


def exact_cells(frame: pd.DataFrame, keys: tuple[str, ...], expected: set, label: str) -> None:
    actual = list(frame.loc[:, list(keys)].itertuples(index=False, name=None))
    missing, extra = expected - set(actual), set(actual) - expected
    duplicates = frame.duplicated(list(keys), keep=False)
    if missing or extra or duplicates.any() or len(actual) != len(expected):
        raise ValueError(f"{label}: expected {len(expected)} unique cells; "
                         f"missing={sorted(missing)[:5]}, extra={sorted(extra)[:5]}, "
                         f"duplicate_rows={int(duplicates.sum())}")


def validate_route_rows(rows: pd.DataFrame, names: list[str]) -> pd.DataFrame:
    """Accept exactly the frozen 20-screen by 10-seed condition-P policy cells."""
    required = {"screen", "policy", "condition", "refused", "set1", "pmid"}
    if required - set(rows.columns):
        raise ValueError(f"route A.1: missing columns {sorted(required - set(rows.columns))}")
    for flag in ("refused", "set1"):
        if not pd.api.types.is_bool_dtype(rows[flag]):
            raise ValueError(f"route A.1: {flag} must contain boolean values")
    selected = ((rows.policy == "task-thompson") & (rows.condition == "P") &
                (rows.screen.isin(names) | rows.set1))
    frame = rows.loc[selected].copy()
    if frame.refused.any() or not frame.set1.all():
        raise ValueError("route A.1: a registered headline row is refused or not marked set1")
    counts = ("seed", "n_genes", "n_hits", "n_picks", "found", "total_hits", "library_size")
    numeric(frame, counts + ("ef",), "route A.1")
    if any((frame[c] != np.floor(frame[c])).any() for c in counts):
        raise ValueError("route A.1: seeds and counts must be integers")
    exact_cells(frame, ("screen", "seed"), set(product(names, SEEDS)), "route A.1")
    if ((frame[list(counts)] < 0).any().any() or (frame.ef < 0).any() or
            (frame.n_picks != BUDGET).any() or (frame.n_genes < BUDGET).any() or
            (frame.n_genes > frame.library_size).any() or
            (frame.n_hits > frame.n_genes).any() or (frame.n_hits > frame.total_hits).any() or
            (frame.total_hits <= 0).any() or (frame.total_hits > frame.library_size).any() or
            (frame.found > frame.n_hits).any() or (frame.found > BUDGET).any()):
        raise ValueError("route A.1: invalid budget, pool counts, hit counts or EF")
    for column in ("n_genes", "n_hits", "total_hits", "library_size", "pmid"):
        if (frame.groupby("screen")[column].nunique(dropna=False) != 1).any():
            raise ValueError(f"route A.1: {column} changes across a screen's seeds")
    if frame.pmid.isna().any() or (frame.pmid.astype(str).str.strip() == "").any():
        raise ValueError("route A.1: publication identifiers are missing")
    expected_ef = frame.found / (BUDGET * frame.total_hits / frame.library_size)
    if not np.allclose(frame.ef, expected_ef, rtol=1e-10, atol=1e-10):
        raise ValueError("route A.1: EF disagrees with the full-library denominator")
    return frame
