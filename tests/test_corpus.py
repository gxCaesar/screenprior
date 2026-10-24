"""The parser must refuse. These tests are the refusal.

The defect this guards against does not drop a row, it relabels one -- a HIT value outside
{YES, NO} read as a non-hit, always in the same direction, invisible downstream. Measured on the
1,225-screen corpus it has never fired, which is exactly why it needs a test rather than a comment.
"""

from __future__ import annotations

import numpy as np
import pytest

from screenprior.corpus import (
    GENEDISCO_SIBLING_PMIDS,
    Corpus,
    ScreenParseError,
    parent_cell_line,
    read_screen,
)

HEADER = "#SCREEN_ID\tIDENTIFIER_ID\tOFFICIAL_SYMBOL\tSCORE.1\tHIT"


def write(tmp_path, *rows, header=HEADER, name="screen.tab.txt"):
    path = tmp_path / name
    path.write_text("\n".join([header, *rows]) + "\n")
    return path


def test_reads_a_well_formed_screen(tmp_path):
    path = write(tmp_path, "1\tA\tTP53\t-1.5\tYES", "1\tB\tEGFR\t0.2\tNO",
                 "1\tC\tMYC\t-\tNO")
    symbols, hits, scores = read_screen(path)
    assert symbols == ["TP53", "EGFR", "MYC"]
    assert list(hits) == [True, False, False]
    assert scores[0] == pytest.approx(-1.5)
    assert np.isnan(scores[2])          # '-' is a stated absence, not a parse failure


def test_an_unexpected_hit_value_stops_the_run(tmp_path):
    """The load-bearing one. 'MAYBE' silently becoming a negative is the defect."""
    path = write(tmp_path, "1\tA\tTP53\t-1.5\tMAYBE")
    with pytest.raises(ScreenParseError, match="would relabel it"):
        read_screen(path)
    _, hits, _ = read_screen(path, strict=False)
    assert list(hits) == [False]        # what the old behaviour did, reachable only on request


def test_a_ragged_row_stops_the_run(tmp_path):
    path = write(tmp_path, "1\tA\tTP53\t-1.5\tYES", "1\tB\tEGFR\tNO")
    with pytest.raises(ScreenParseError, match="fields, header has"):
        read_screen(path)


def test_an_empty_symbol_stops_the_run(tmp_path):
    path = write(tmp_path, "1\tA\t\t-1.5\tYES")
    with pytest.raises(ScreenParseError, match="empty OFFICIAL_SYMBOL"):
        read_screen(path)


def test_an_unparseable_score_stops_the_run(tmp_path):
    path = write(tmp_path, "1\tA\tTP53\tn/a\tYES")
    with pytest.raises(ScreenParseError, match="SCORE.1 is"):
        read_screen(path)


def test_a_missing_column_stops_the_run(tmp_path):
    path = write(tmp_path, "1\tA\tTP53\t-1.5", header="#SCREEN_ID\tIDENTIFIER_ID\tOFFICIAL_SYMBOL\tSCORE.1")
    with pytest.raises(ScreenParseError, match="missing column"):
        read_screen(path)


def test_undecodable_bytes_stop_the_run(tmp_path):
    path = tmp_path / "bad.tab.txt"
    path.write_bytes(HEADER.encode() + b"\n1\tA\tTP\xff53\t-1.5\tYES\n")
    with pytest.raises(ScreenParseError, match="not valid UTF-8"):
        read_screen(path)


# --------------------------------------------------------------------------- Corpus

def toy() -> Corpus:
    hit = np.array([[1, 0, 1], [0, 1, 0], [1, 1, 0], [0, 0, 0]], dtype=bool)
    tested = np.ones_like(hit)
    tested[3, 2] = False
    return Corpus(genes=np.array(["A", "B", "C", "D"]),
                  screen_id=np.array(["10", "11", "12"]),
                  pmid=np.array(["p1", "p1", "p2"]),
                  cell_line=np.array(["K562", "HeLa", "K562"]),
                  phenotype=np.array(["prolif", "prolif", "virus"]),
                  hit=hit, tested=tested)


def test_a_hit_on_an_untested_gene_is_rejected_at_construction():
    hit = np.ones((2, 1), dtype=bool)
    tested = np.zeros((2, 1), dtype=bool)
    with pytest.raises(ValueError, match="did not test"):
        Corpus(np.array(["A", "B"]), np.array(["1"]), np.array(["p"]), np.array(["K562"]),
               np.array(["x"]), hit, tested)


def test_promiscuity_divides_by_screens_that_tested_the_gene():
    corpus = toy()
    # Gene D is a hit nowhere and was not tested by screen 12: 0 / 2, not 0 / 3.
    assert corpus.promiscuity[3] == pytest.approx(0.0)
    assert corpus.promiscuity[2] == pytest.approx(2 / 3)


def test_donors_exclude_the_target_publication_not_merely_the_target_screen():
    corpus = toy()
    # Screens 0 and 1 share pmid p1, so a target of 0 must not see 1 either.
    assert list(corpus.donors_for(0)) == [2]
    assert list(corpus.donors_for(2)) == [0, 1]


def test_cell_line_blocking_removes_more():
    corpus = toy()
    assert list(corpus.donors_for(2)) == [0, 1]
    assert list(corpus.donors_for(2, block_cell_line=True)) == [1]   # screen 0 is also K562


def test_parent_cell_lines_merge_sublines_and_spellings():
    """Blocking on raw strings left HEK293T among HEK-293's donors."""
    cases = {"HEK293T": "HEK293", "HEK-293": "HEK293", "HEK293T-CRISPRa": "HEK293",
             "A549-ACE2": "A549", "A2780cis": "A2780", "K-562": "K562",
             "CD4+ T-cells": "TCELL", "Regulatory T cell": "TCELL", "hTERT-RPE1": "HTERTRPE1",
             "HeLa": "HELA"}
    assert {name: parent_cell_line(name) for name in cases} == cases


def test_parent_blocking_removes_a_subline_that_raw_blocking_keeps():
    corpus = toy()
    corpus = Corpus(corpus.genes, corpus.screen_id, np.array(["p1", "p2", "p3"]),
                    np.array(["HEK293T", "HEK-293", "K562"]), corpus.phenotype, corpus.hit,
                    corpus.tested)
    assert list(corpus.donors_for(0, block_cell_line=True)) == [1, 2]
    parent = corpus.with_parent_cell_lines()
    assert list(parent.donors_for(0, block_cell_line=True)) == [2]
    assert list(parent.donors_for(0)) == [1, 2]          # publication blocking is unchanged


def test_no_donors_left_raises_rather_than_returning_an_empty_set():
    corpus = toy()
    single = Corpus(corpus.genes, corpus.screen_id[:1], corpus.pmid[:1], corpus.cell_line[:1],
                    corpus.phenotype[:1], corpus.hit[:, :1], corpus.tested[:, :1])
    with pytest.raises(ValueError, match="no donors remain"):
        single.donors_for(0)


def test_one_target_per_publication_and_the_choice_is_deterministic():
    corpus = toy()
    targets = corpus.targets_one_per_publication()
    assert len(targets) == len({p for p in corpus.pmid})
    assert list(targets) == list(corpus.targets_one_per_publication())


def test_dropping_publications_removes_every_screen_they_contributed():
    corpus = toy()
    smaller = corpus.drop_pmids(["p1"])
    assert list(smaller.screen_id) == ["12"]
    assert smaller.hit.shape[1] == 1


def test_the_genedisco_siblings_are_named_not_inferred():
    assert GENEDISCO_SIBLING_PMIDS == {"35113687", "31921143", "33574281"}


def test_round_trip_through_disk_preserves_everything(tmp_path):
    corpus = toy()
    corpus.save(tmp_path / "c.npz")
    back = Corpus.load(tmp_path / "c.npz")
    assert np.array_equal(back.hit, corpus.hit)
    assert np.array_equal(back.tested, corpus.tested)
    assert list(back.pmid) == list(corpus.pmid)
