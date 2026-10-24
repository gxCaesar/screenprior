"""Gene representations, split by whether they depend on which donors are blocked.

The split is not tidiness, it is a leak boundary.

  STATIC     DepMap co-essentiality. Computed once, identical for every target screen, because it
             does not involve the corpus at all.
  PER-TARGET The corpus embedding -- an SVD of the DONOR hit matrix. The donor set changes with the
             target, because blocking removes the target's own publication. Computing this once
             over the whole corpus would put the target screen's own hit pattern into the gene
             embedding that ranks it, which is answer-key leakage arriving through a factorisation.

`corpus_embedder` therefore returns a callable rather than an array for the corpus half, and
`harness.run_screen` calls it after blocking, so the embedding uses only eligible donors.

The static half is `depmap_matrix` followed by `static_block`: read the gene-effect release
with genes as rows and missing values as 0.0, sort by symbol, keep the corpus genes DepMap
measures, and take a rank-64
randomized SVD (four power iterations, random state 0) scaled by its singular values. The same
function returns the mean gene effect over those rows, which defines the pan-essential cut.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.utils.extmath import randomized_svd

__all__ = ["STATIC_RANK", "align_to", "corpus_embedder", "depmap_matrix", "static_block",
           "svd_embedding"]

DEPMAP_FILENAMES = ("CRISPRGeneEffect.csv", "CRISPR_gene_effect.csv", "gene_effect.csv",
                    "ceres-gene-effects.csv")

# The rank of the static block in every reported run.
STATIC_RANK = 64


def depmap_matrix(source) -> tuple:
    """Return (matrix, symbols) with genes as ROWS and sorted by symbol, whichever layout the
    release uses. `source` is the gene-effect file itself or a folder holding one.

    DepMap has shipped at least three layouts across releases: genes as rows with CCLE names as
    columns (17Q4-era `ceres-gene-effects.csv`), cell lines as rows with `SYMBOL (ENTREZ)` columns
    (`gene_effect.csv`, `CRISPR_gene_effect.csv`), and ModelID rows (`CRISPRGeneEffect.csv`).
    Guessing wrong transposes the matrix silently and every downstream number is then a fact about
    cell lines rather than genes, so the layout is decided by the filename rather than by shape, and
    a file whose name is not one of these is refused rather than read as whichever layout a renamed
    file would imply. Missing values become 0.0, as in the reported runs; 21Q4 has 3,984 of them
    across 1,054 cell lines, in 664 genes, 663 of which are among the 17,268 the corpus shares.
    """
    path = Path(source)
    if path.is_dir():
        found = next((path / name for name in DEPMAP_FILENAMES if (path / name).exists()), None)
        if found is None:
            raise FileNotFoundError(f"no DepMap gene-effect file in {path}; expected one of "
                                    f"{DEPMAP_FILENAMES}")
        path = found
    elif path.name not in DEPMAP_FILENAMES:
        raise ValueError(f"{path.name} is not one of {DEPMAP_FILENAMES}. The layout (genes as rows "
                         "or as columns) is decided by the filename, so keep the name DepMap gave "
                         "the file")
    frame = pd.read_csv(path, index_col=0)
    if path.name == "ceres-gene-effects.csv":
        frame = frame.loc[~frame.index.duplicated()]
        values = np.nan_to_num(frame.to_numpy(dtype=np.float32), nan=0.0)
        symbols = np.array([str(g).split(" (")[0].strip().upper() for g in frame.index])
    else:
        frame.columns = [str(c).split(" (")[0].strip().upper() for c in frame.columns]
        frame = frame.loc[:, ~frame.columns.duplicated()]
        values = np.nan_to_num(frame.to_numpy(dtype=np.float32), nan=0.0).T
        symbols = np.array(list(frame.columns))
    # SORTED BY SYMBOL, because callers index this with np.searchsorted, which returns silently
    # wrong rows on an unsorted array rather than raising. The 21Q4 columns are sorted as written;
    # upper-cased, two pairs fall out of order (C1S before C1ORF100, CXXC5 before CXORF38), and
    # sorting then moves 60 of the 17,386 symbols. A stable sort, so that of two labels cleaning to
    # one symbol the first in file order is the one kept.
    order = np.argsort(symbols, kind="stable")
    symbols, values = symbols[order], values[order]
    first = np.ones(len(symbols), dtype=bool)
    first[1:] = symbols[1:] != symbols[:-1]
    if not first.all():
        symbols, values = symbols[first], values[first]
    return values, symbols


def svd_embedding(matrix: np.ndarray, rank: int = 64, seed: int = 0) -> np.ndarray:
    """Left singular vectors scaled by their singular values."""
    u, s, _ = randomized_svd(matrix.astype(np.float32), n_components=rank, n_iter=4,
                             random_state=seed)
    return (u * s).astype(np.float32)


def corpus_embedder(rank: int = 64, seed: int = 0):
    """A callable (donor_hit_matrix) -> (n_genes, rank), to be applied AFTER blocking.

    Returned as a callable on purpose: an array here would be computed once over every screen,
    including the target's, and the target's own hit pattern would enter the embedding that ranks
    it. See this module's docstring.
    """
    def embed(donor_hits: np.ndarray) -> np.ndarray:
        return svd_embedding(donor_hits.astype(np.float32), rank=rank, seed=seed)
    return embed


def _require_sorted(name: str, labels: np.ndarray) -> None:
    """Refuse labels that np.searchsorted cannot index correctly.

    searchsorted on an array that is not sorted, or that repeats a label, returns a valid-looking
    index to the wrong row instead of raising, so the check has to be made before it is called.
    """
    if len(labels) > 1 and not np.all(labels[:-1] < labels[1:]):
        raise ValueError(f"the {name} are not sorted and unique; np.searchsorted would return "
                         "wrong rows without raising, so nothing is aligned")


def align_to(genes: np.ndarray, values: np.ndarray, symbols: np.ndarray) -> tuple:
    """Restrict both sides to the genes they share, in the corpus's order.

    Returns (row indices into `genes`, the aligned values). Raises if either label array is not
    sorted and unique, and if the overlap is implausibly small, because an empty or tiny
    intersection is almost always a symbol-namespace mismatch rather than biology, and silently
    proceeding produces a table of noise.
    """
    genes, symbols = np.asarray(genes), np.asarray(symbols)
    _require_sorted("corpus genes", genes)
    _require_sorted("feature-source symbols", symbols)
    shared = np.intersect1d(genes, symbols)
    if len(shared) < 0.5 * min(len(genes), len(symbols)):
        raise ValueError(
            f"only {len(shared):,} genes shared between the corpus ({len(genes):,}) and the "
            f"feature source ({len(symbols):,}); this is usually a symbol-namespace mismatch")
    return np.searchsorted(genes, shared), values[np.searchsorted(symbols, shared)]


def static_block(genes: np.ndarray, values: np.ndarray, symbols: np.ndarray,
                 rank: int = STATIC_RANK, seed: int = 0) -> tuple:
    """The static block and the mean gene effect for the corpus genes the source measures.

    `values` and `symbols` are what `depmap_matrix` returns. Returns (rows, block, effect): `rows`
    indexes the corpus genes that DepMap also measures, in the corpus's order; `block` is the
    rank-`rank` SVD embedding of those genes' rows, (len(rows), rank) float32; `effect` is each
    such gene's mean over the cell lines, the value the pan-essential cut is applied to.

    A corpus gene DepMap does not measure is dropped, not given a row of zeros, so the corpus a run
    reads has to be restricted to `rows` as well; `screenprior build-static` writes both. On the
    ORCS 2.0.18 corpus and 21Q4 that keeps 17,268 of 81,696 genes.
    """
    rows, aligned = align_to(genes, values, symbols)
    if not 0 < rank <= min(aligned.shape):
        raise ValueError(f"rank {rank} is not between 1 and the smaller side of the aligned "
                         f"matrix {aligned.shape}; the SVD would return fewer columns than asked")
    return rows, svd_embedding(aligned, rank=rank, seed=seed), aligned.mean(axis=1)
