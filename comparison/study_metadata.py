"""Metadata checks for the reported 20-screen comparison and its nested subset."""
from __future__ import annotations

import numpy as np
import pandas as pd
from row_validation import BUDGET, F2_SIZE, numeric

# The reported order fixes the finite-sample bootstrap draws.
SCREEN_ORDER = ('1985', 'U_1736_dec', '2462', '1953', '2054', '1900', 'U_2423_merged', 'U_1733_merged', 'U_1987_merged', 'U_2422_merged', 'U_2427_dec', '2076', '1830', '2466', 'U_1863_inc', '2060', '1827', 'U_1735_dec', '1905', '2404')
NEVER_READ_SCREENS = frozenset(('U_1736_dec', '2462', '2054', 'U_2423_merged', 'U_2422_merged', 'U_2427_dec', '1830', 'U_1863_inc', '1827', 'U_1735_dec', '1905'))


def validate_metadata(metadata: pd.DataFrame) -> pd.DataFrame:
    """Keep the reported order, publication groups and development-exposure membership."""
    frame = metadata.copy(deep=True)
    columns = ("screen_order", "screen", "publication_group", "development_read", "never_read",
               "library_size", "full_library_hits", "matched_pool_n", "matched_pool_h",
               "native_pool_n", "native_pool_h")
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(f"screen metadata: missing columns {sorted(missing)}")
    if len(frame) != 20 or tuple(frame.screen) != SCREEN_ORDER:
        raise ValueError("screen metadata: expected the reported 20 screens in their original order")
    for col in ("screen", "publication_group"):
        if frame[col].isna().any() or frame[col].map(lambda v: not str(v) or str(v) != str(v).strip()).any():
            raise ValueError(f"screen metadata: missing or untrimmed {col}")
        frame[col] = frame[col].astype(str)
    counts = ("screen_order", "library_size", "full_library_hits", "matched_pool_n",
              "matched_pool_h", "native_pool_n", "native_pool_h")
    numeric(frame, counts, "screen metadata")
    if any((frame[c] != np.floor(frame[c])).any() for c in counts):
        raise ValueError("screen metadata: pool and order counts must be integers")
    if not np.array_equal(frame.screen_order.to_numpy(), np.arange(1, 21)):
        raise ValueError("screen metadata: screen_order must be 1 through 20 in table order")
    for flag in ("development_read", "never_read"):
        if not pd.api.types.is_bool_dtype(frame[flag]):
            raise ValueError(f"screen metadata: {flag} must contain boolean values")
    if (frame.development_read == frame.never_read).any():
        raise ValueError("screen metadata: never_read must be the complement of development_read")
    if frozenset(frame.loc[frame.never_read, "screen"]) != NEVER_READ_SCREENS:
        raise ValueError("screen metadata: the reported nested never-read membership changed")
    if frame.publication_group.nunique() != 15 or frame.loc[frame.never_read, "publication_group"].nunique() != 7:
        raise ValueError("screen metadata: expected 15 publication groups and seven in the nested subset")
    if ((frame.matched_pool_n < BUDGET).any() or
            (frame.matched_pool_n > frame.library_size).any() or
            (frame.native_pool_n != F2_SIZE).any() or
            (frame.matched_pool_h < 0).any() or
            (frame.matched_pool_h > frame.matched_pool_n).any() or
            (frame.matched_pool_h > frame.native_pool_h).any() or
            (frame.native_pool_h > frame.full_library_hits).any() or
            (frame.native_pool_h > frame.native_pool_n).any() or
            (frame.full_library_hits <= 0).any() or
            (frame.full_library_hits > frame.library_size).any()):
        raise ValueError("screen metadata: inconsistent candidate-pool or full-library counts")
    return frame


def audit_metadata_counts(route: pd.DataFrame, assayformer: pd.DataFrame,
                          metadata: pd.DataFrame) -> None:
    """Check both methods' saved pool/library scalars against the public screen table."""
    reference = metadata.set_index("screen")
    for frame, pairs in (
        (route, (("n_genes", "matched_pool_n"), ("n_hits", "matched_pool_h"),
                 ("library_size", "library_size"), ("total_hits", "full_library_hits"))),
        (assayformer.loc[assayformer.pool == "matched"],
         (("n_genes", "matched_pool_n"), ("n_hits", "matched_pool_h"),
          ("library_size", "library_size"), ("total_hits", "full_library_hits"))),
        (assayformer.loc[assayformer.pool == "native"],
         (("n_genes", "native_pool_n"), ("n_hits", "native_pool_h"),
          ("library_size", "library_size"), ("total_hits", "full_library_hits"))),
    ):
        for saved, expected in pairs:
            if not np.array_equal(frame[saved].to_numpy(), reference.loc[frame.screen, expected].to_numpy()):
                raise ValueError(f"saved rows: {saved} differs from screen metadata {expected}")
    if not np.array_equal(route.pmid.to_numpy(), reference.loc[route.screen, "publication_group"].to_numpy()):
        raise ValueError("saved rows: publication groups differ from screen metadata")
