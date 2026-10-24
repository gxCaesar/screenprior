"""The signed-rank rule used by the reported comparison, with an explicit exact/normal choice."""
from __future__ import annotations

import numpy as np
from scipy import stats


def signed_rank_p(d, zero_method: str = "wilcox") -> float:
    """Two-sided Wilcoxon signed-rank P, with the exact-or-normal choice made here, not by scipy.

    scipy's `method="auto"` has not meant the same thing in every release. 1.14 takes the exact
    null distribution for n <= 50 even when the |d| contain ties, and that distribution assumes
    there are none; 1.18, the release the verification gate ran under when the manuscript's values
    were checked, takes the normal approximation with the tie correction instead. Measured on the
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
