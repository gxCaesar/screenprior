"""The static block and the mean gene effect, against values derived by hand.

These two arrays feed every reported number: the block is half of each policy's representation and
the mean gene effect decides which genes may be acquired at all. The failures guarded here are the
silent ones: a symbol axis that is almost sorted, a file read in the wrong orientation, a missing
value averaged differently, and a join that drops most genes without saying so.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from screenprior.features import align_to, depmap_matrix, static_block


def write_depmap(folder, columns: dict, name: str = "CRISPR_gene_effect.csv"):
    """A gene-effect file in the 21Q4 layout: cell lines as rows, `SYMBOL (ENTREZ)` columns, in
    the order the dict gives them."""
    n_lines = len(next(iter(columns.values())))
    frame = pd.DataFrame(columns, index=[f"ACH-{i:06d}" for i in range(n_lines)])
    frame.index.name = "DepMap_ID"
    path = folder / name
    frame.to_csv(path)
    return path


def signed(block: np.ndarray) -> np.ndarray:
    """Each column multiplied by the sign of its largest-magnitude entry. A singular vector is
    defined up to sign, so this is the form in which two SVDs can be compared."""
    block = np.asarray(block, dtype=np.float64)
    pivot = block[np.abs(block).argmax(axis=0), np.arange(block.shape[1])]
    return block * np.sign(pivot)


def test_static_block_matches_a_hand_computed_svd_and_mean(tmp_path):
    """Genes A-D measured in three cell lines; E is in the corpus only and Z in DepMap only.

    After the missing value in D becomes 0.0 the gene-by-line matrix is
        A [2, 2, 0]   B [1, -1, 0]   C [0, 0, 0]   D [0, 0, 3]
    so M^T M = [[5, 3, 0], [3, 5, 0], [0, 0, 9]], with eigenvalues 9, 8 and 2 on the right singular
    vectors (0, 0, 1), (1, 1, 0)/sqrt2 and (1, -1, 0)/sqrt2. U*S = M V, so the rank-2 block is
    3 on D in the first column, 4/sqrt2 = 2 sqrt2 on A in the second, and zero elsewhere. The means
    are 4/3, 0, 0 and 1 -- not 1.5 for D, which is what skipping the missing value would give.
    """
    path = write_depmap(tmp_path, {"D (4)": [np.nan, 0, 3], "B (2)": [1, -1, 0],
                                   "Z (26)": [5, 5, 5], "A (1)": [2, 2, 0], "C (3)": [0, 0, 0]})
    values, symbols = depmap_matrix(path)
    assert symbols.tolist() == ["A", "B", "C", "D", "Z"]
    corpus_genes = np.array(["A", "B", "C", "D", "E"])
    rows, block, effect = static_block(corpus_genes, values, symbols, rank=2)

    assert rows.tolist() == [0, 1, 2, 3]
    assert block.dtype == np.float32 and effect.dtype == np.float32
    expected = np.array([[0, 2 * np.sqrt(2)], [0, 0], [0, 0], [3, 0]])
    np.testing.assert_allclose(signed(block), expected, atol=1e-5)
    np.testing.assert_allclose(effect, [4 / 3, 0, 0, 1], rtol=1e-6)


def test_static_block_agrees_with_an_exact_svd(tmp_path):
    """On a matrix with distinct singular values the randomized SVD must agree with LAPACK's exact
    one: with eight cell lines, a rank of three and ten oversamples the sketch spans every column,
    so any disagreement beyond float32 rounding is a defect, not an approximation."""
    rng = np.random.default_rng(0)
    genes = [f"G{i:02d}" for i in range(30)]
    matrix = rng.normal(size=(30, 8))
    path = write_depmap(tmp_path, {f"{g} ({i})": matrix[i] for i, g in enumerate(genes)})
    values, symbols = depmap_matrix(path)
    rows, block, _ = static_block(np.array(genes), values, symbols, rank=3)

    u, s, _ = np.linalg.svd(matrix.astype(np.float32).astype(np.float64), full_matrices=False)
    assert rows.tolist() == list(range(30))
    np.testing.assert_allclose(signed(block), signed(u[:, :3] * s[:3]), atol=1e-4)


def test_symbols_inverted_by_upper_casing_are_sorted_rather_than_misaligned(tmp_path):
    """The 21Q4 file lists C1S before C1orf100 and CXXC5 before CXorf38. Upper-cased, both pairs
    are out of order, and np.searchsorted on that axis returns another gene's row without raising.
    Each gene here is a constant, so its mean names the column it was read from."""
    in_file_order = ["A1BG (1)", "C1S (716)", "C1orf100 (200159)", "CXXC5 (51523)",
                     "CXorf38 (159013)", "ZZZ3 (26009)"]
    frame = {name: [float(k + 1)] * 4 for k, name in enumerate(in_file_order)}
    path = write_depmap(tmp_path, frame)
    values, symbols = depmap_matrix(path)
    corpus_genes = np.array(sorted(n.split(" (")[0].upper() for n in in_file_order))
    _, _, effect = static_block(corpus_genes, values, symbols, rank=1)

    truth = {n.split(" (")[0].upper(): frame[n][0] for n in in_file_order}
    assert effect.tolist() == [truth[g] for g in corpus_genes]

    # The same lookup on the unsorted axis lands on the wrong gene, so the check above has teeth.
    unsorted = np.array([n.split(" (")[0].upper() for n in in_file_order])
    landed = unsorted[np.minimum(np.searchsorted(unsorted, corpus_genes), len(unsorted) - 1)]
    assert (landed != corpus_genes).any()


def test_the_layout_is_decided_by_the_filename(tmp_path):
    """The same numbers written genes-as-rows (17Q4 name) and lines-as-rows (21Q4 name) must come
    back as the same gene-by-line matrix."""
    data = {"B (2)": [1.0, 2.0, 3.0], "A (1)": [4.0, 5.0, 6.0]}
    lines_as_rows = tmp_path / "new"
    lines_as_rows.mkdir()
    write_depmap(lines_as_rows, data)
    genes_as_rows = tmp_path / "old"
    genes_as_rows.mkdir()
    pd.DataFrame(data, index=["L1", "L2", "L3"]).T.to_csv(genes_as_rows
                                                         / "ceres-gene-effects.csv")
    new_values, new_symbols = depmap_matrix(lines_as_rows)
    old_values, old_symbols = depmap_matrix(genes_as_rows)
    assert new_symbols.tolist() == old_symbols.tolist() == ["A", "B"]
    np.testing.assert_array_equal(new_values, [[4, 5, 6], [1, 2, 3]])
    np.testing.assert_array_equal(old_values, new_values)


def test_a_renamed_or_missing_release_is_refused(tmp_path):
    """A file under another name could be either orientation, so it is not read at all."""
    path = write_depmap(tmp_path, {"A (1)": [1.0, 2.0]}, name="depmap_21q4.csv")
    with pytest.raises(ValueError, match="decided by the filename"):
        depmap_matrix(path)
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileNotFoundError, match="no DepMap gene-effect file"):
        depmap_matrix(empty)


@pytest.mark.parametrize("genes, symbols", [
    (["B", "A", "C"], ["A", "B", "C"]),       # corpus genes out of order
    (["A", "B", "C"], ["A", "C", "B"]),       # source symbols out of order
    (["A", "B", "C"], ["A", "B", "B", "C"]),  # a repeated symbol
])
def test_unsorted_or_repeated_labels_are_refused(genes, symbols):
    values = np.arange(len(symbols), dtype=np.float32)[:, None]
    with pytest.raises(ValueError, match="not sorted and unique"):
        align_to(np.array(genes), values, np.array(symbols))


def test_an_implausibly_small_overlap_is_refused():
    """Two of ten genes shared is a namespace mismatch (Ensembl ids against symbols, say), not a
    smaller corpus."""
    genes = np.array([f"G{i}" for i in range(10)])
    symbols = np.array(sorted(["G0", "G1"] + [f"ENSG{i:011d}" for i in range(8)]))
    with pytest.raises(ValueError, match="symbol-namespace mismatch"):
        align_to(genes, np.zeros((10, 3), dtype=np.float32), symbols)


def test_a_rank_the_matrix_cannot_supply_is_refused():
    """randomized_svd returns fewer columns than asked when the rank exceeds the matrix's smaller
    side, and the block would then be narrower than the one the runs used."""
    genes = np.array(["A", "B", "C", "D"])
    with pytest.raises(ValueError, match="fewer columns than asked"):
        static_block(genes, np.ones((4, 3), dtype=np.float32), genes, rank=4)
