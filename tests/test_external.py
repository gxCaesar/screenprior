"""Synthetic checks of the external-target boundary, frozen slots and label-free controls."""

from __future__ import annotations

import numpy as np
import pytest

from screenprior import external
from screenprior.corpus import Corpus
from screenprior.harness import RunConfig, run_screen
from screenprior.policies import POLICIES
from screenprior.random_control import continuous_random_indices


def base_corpus(n=8):
    """Two small donors, one donor with zero tested genes for a candidate, and a blocked sibling."""
    hit = np.zeros((n, 4), dtype=bool)
    tested = np.ones_like(hit)
    if n == 8:
        hit[[0, 1, 4], 0] = True
        hit[[1, 3, 6], 1] = True
        tested[:, 0] = False
        tested[[0, 1, 2, 3, 4, 5], 0] = True
        tested[:, 1] = False
        tested[[0, 1, 2, 3, 6], 1] = True
    else:
        hit[np.arange(n) % 13 == 0, 0] = True
        hit[np.arange(n) % 17 == 0, 1] = True
    hit[:, 2:] = True
    return Corpus(np.array([f"G{i:05d}" for i in range(n)]),
                  np.array(["d1", "d2", "d3", "d4"]),
                  np.array(["P1", "P2", "BLOCK", "BLOCK"]),
                  np.array(["L1", "L2", "L3", "L4"]),
                  np.array(["donor"] * 4), hit, tested)


def append_target(base, hit=None, pub="EXT", sid="external"):
    labels = np.arange(len(base.genes)) % 11 == 0 if hit is None else hit
    return external.external_corpus(
        base, np.ones(len(base.genes), dtype=bool), labels, pub, sid,
        {"cell_line": "EXT-LINE", "phenotype": "synthetic"}, {"BLOCK"})


def test_external_construction_blocks_entire_publication_and_appends_only_one_target():
    """A previous external target cannot become a donor for the next target."""
    base = base_corpus()
    first = append_target(base)
    second = append_target(base, pub="OTHER", sid="other")
    assert first.screen_id.tolist() == ["d1", "d2", "external"]
    assert second.screen_id.tolist() == ["d1", "d2", "other"]
    assert base.screen_id.tolist() == ["d1", "d2", "d3", "d4"]
    np.testing.assert_array_equal(first.hit[:, :2], base.hit[:, :2])
    np.testing.assert_array_equal(first.tested[:, :2], base.tested[:, :2])
    assert first.donors_for(2).tolist() == [0, 1]
    with pytest.raises(ValueError, match="already in the supplied base"):
        append_target(first, pub="THIRD", sid="third")


def test_external_identity_and_invalid_mask_are_refused():
    """Refuse in-base targets, untested hits, absent donors and missing block decisions."""
    base = base_corpus()
    tested = np.ones(8, dtype=bool)
    hit = np.arange(8) == 0
    for pub, sid, meta, blocks, match in [
        ("P1", "external", {}, set(), "already in the base"),
        ("new", "d1", {}, set(), "already in the base"),
        ("pmid_P1", "external", {"pmid": "P1"}, set(), "already in the base"),
        ("new", "external", {}, {"P1", "P2", "BLOCK"}, "no donor"),
        ("new", "external", {}, {"typo"}, "unknown blocked"),
        ("new", "external", {}, None, "explicit blocked"),
    ]:
        with pytest.raises(ValueError, match=match):
            external.external_corpus(base, tested, hit, pub, sid, meta, blocks)
    tested[0] = False
    with pytest.raises(ValueError, match="untested"):
        external.external_corpus(base, tested, hit, "new", "external", {}, set())
    with pytest.raises(ValueError, match="binary vector"):
        external.external_corpus(base, np.ones(7), hit, "new", "external", {}, set())


def test_fixed_genedisco_exclusion_is_not_reintroduced_by_an_empty_author_block():
    """The author-off sensitivity still preserves the frozen GeneDisco sibling exclusion."""
    base = base_corpus()
    base.pmid = base.pmid.astype("U16")
    base.pmid[0] = "33574281"
    assert base.pmid[0] == "33574281"
    result = external.external_corpus(base, np.ones(8), np.arange(8) == 0,
                                      "EXT", "external", {}, set())
    assert "33574281" not in result.pmid


def test_label_free_pure_controls_have_hand_computed_denominators_and_stable_ties(monkeypatch):
    """Gene-specific donor coverage, rather than donor count, determines corpus frequency."""
    base = base_corpus()
    labels = np.isin(np.arange(8), [0, 4, 6])
    corpus = append_target(base, labels)
    ge = np.array([-3, -1, -2, 0, -4, 3, 4, 5], dtype=float)

    def forbid_core(*args, **kwargs):
        raise AssertionError("label-free baselines invoked the sequential fitting core")

    monkeypatch.setattr(external, "run_screen", forbid_core)
    monkeypatch.setattr(external, "RunConfig", forbid_core)
    result = external.label_free_baselines(
        corpus, 2, ge, pub_id="EXT", budget=4, batch=2, seeds=(0, 1))
    assert len(result["rows"]) == 6
    pure_rows = [r for r in result["rows"] if r["seed"] == -1]
    assert len(pure_rows) == 2
    assert {r["cycle"] for r in pure_rows} == {0}
    traces = {t["policy"]: t for t in result["traces"] if t["seed"] == -1}
    np.testing.assert_array_equal(traces["corpus-frequency/pure"]["picks"], [1, 4, 6, 0])
    np.testing.assert_array_equal(traces["depmap-essentiality/pure"]["picks"], [4, 0, 2, 1])
    outcomes = {r["policy"]: r for r in pure_rows}
    assert outcomes["corpus-frequency/pure"]["acquired_hits"] == 3
    assert outcomes["corpus-frequency/pure"]["recall"] == 1
    assert outcomes["depmap-essentiality/pure"]["recall"] == 2 / 3
    assert all(r["chance"] == 0.5 and r["n_genes"] == 8 and r["n_hits"] == 3
               and r["acquired"] == 4 for r in pure_rows)


def test_shared_opening_baselines_use_the_core_template_and_never_target_labels():
    """Changing external labels changes scoring only, not either control's acquisitions."""
    base = base_corpus()
    left = append_target(base, np.isin(np.arange(8), [0, 4, 6]))
    right = append_target(base, np.isin(np.arange(8), [1, 3]))
    ge = np.arange(8, dtype=float)
    a = external.label_free_baselines(left, 2, ge, pub_id="EXT", budget=4, batch=2, seeds=(2,))
    b = external.label_free_baselines(right, 2, ge, pub_id="EXT", budget=4, batch=2, seeds=(2,))
    for at, bt in zip(a["traces"], b["traces"]):
        np.testing.assert_array_equal(at["picks"], bt["picks"])
        assert len(np.unique(at["picks"])) == 4
        if at["seed"] == 2:
            np.testing.assert_array_equal(at["picks"][:2],
                                          np.random.default_rng(2).choice(8, 2, replace=False))


def test_nonessential_baseline_pool_uses_the_same_cut_and_global_gene_indices():
    """A score ranked over an untrimmed pool cannot pass this non-contiguous candidate test."""
    corpus = append_target(base_corpus(), np.isin(np.arange(8), [0, 4, 6]))
    ge = np.array([-3, -1, -2, 0, -4, 3, 4, 5], dtype=float)
    result = external.label_free_baselines(
        corpus, 2, ge, pub_id="EXT", pool="non-essential", budget=2, batch=1, seeds=())
    assert len(result["rows"]) == 2
    assert all(r["n_genes"] == 4 and r["n_hits"] == 1 for r in result["rows"])
    traces = {t["policy"]: t["picks"] for t in result["traces"]}
    np.testing.assert_array_equal(traces["corpus-frequency/pure"], [6, 3])
    np.testing.assert_array_equal(traces["depmap-essentiality/pure"], [3, 5])


class StubModel:
    """Cheap deterministic predictions, used only to compare actual core policy RNG paths."""
    def __init__(self):
        self.fits = 0

    def tune(self, X, y, seed):
        return 10.0

    def fit_predict(self, X, y, chosen, alpha, seed):
        self.fits += 1
        mean = np.arange(len(X), dtype=float) / len(X)
        return mean, 0.1 + mean * 0.05


def test_frozen_core_slots_rng_and_blocked_embedding_are_preserved_with_stub_model():
    """Only random's continuation changes; all eight other core trace paths must remain exact."""
    base = base_corpus(n=5000)
    corpus = append_target(base)
    features, ge = np.zeros((5000, 1)), np.zeros(5000)
    seen = []

    def embed(donor_hits):
        seen.append(donor_hits.copy())
        return np.zeros((5000, 1), dtype=np.float32)

    model = StubModel()
    original_registry = dict(POLICIES)
    wrapped = external.run_one_target(corpus, 2, features, ge, pub_id="EXT",
                                      embed_donors=embed, model=model)
    assert wrapped["status"] == "complete"
    assert len(seen) == 1
    np.testing.assert_array_equal(seen[0], base.hit[:, :2])
    assert len(wrapped["rows"]) == 900 and len(wrapped["traces"]) == 90
    assert model.fits == 8 * 9 * 10
    assert set(POLICIES) == set(original_registry)
    assert all(POLICIES[name] is value for name, value in original_registry.items())
    controls = external.label_free_baselines(corpus, 2, ge, pub_id="EXT")
    assert len(controls["rows"]) == len(controls["traces"]) == 22
    assert sum(row["seed"] == -1 for row in controls["rows"]) == 2
    assert model.fits == 8 * 9 * 10
    for trace in controls["traces"]:
        if trace["seed"] != -1:
            opening = next(t["picks"][:64] for t in wrapped["traces"]
                           if t["seed"] == trace["seed"])
            np.testing.assert_array_equal(trace["picks"][:64], opening)
    config = RunConfig(seeds=external.SEEDS, pool="full", model=StubModel(),
                       record_cycles=True, record_picks=True)
    _, reference = run_screen(corpus, 2, features, ge, policies=external.CORE_POLICIES,
                               config=config, embed_donors=lambda h: np.zeros((5000, 1)))
    for actual, frozen in zip(wrapped["traces"], reference):
        assert (actual["seed"], actual["policy"]) == (frozen["seed"], frozen["policy"])
        if actual["policy"] != "random":
            np.testing.assert_array_equal(actual["picks"], frozen["picks"])
        else:
            expected = continuous_random_indices(5000, actual["seed"])
            np.testing.assert_array_equal(actual["picks"], expected)
        np.testing.assert_array_equal(actual["picks"][:64], frozen["picks"][:64])
    assert len(wrapped["diagnostics"]) == 10
    for row in wrapped["rows"]:
        trace = next(t for t in wrapped["traces"]
                     if t["seed"] == row["seed"] and t["policy"] == row["policy"])
        exact = int(corpus.hit[trace["picks"][:row["acquired"]], 2].sum())
        assert row["acquired_hits"] == exact and row["recall"] == exact / row["n_hits"]


def test_cycle3_diagnostic_is_post_acquisition_and_does_not_draw_rng(monkeypatch):
    """Only cycle 3 reveals the matching donor; a pre-cycle statistic would stay at two donors."""
    n = 640
    hit = np.zeros((n, 2), dtype=bool)
    hit[128:160, 0] = True
    hit[:32, 1] = True
    base = Corpus(np.array([f"G{i}" for i in range(n)]), np.array(["D0", "D1"]),
                  np.array(["P0", "P1"]), np.array(["L0", "L1"]),
                  np.array(["donor", "donor"]), hit, np.ones_like(hit))
    labels = np.zeros(n, dtype=bool)
    labels[128:160] = True
    corpus = external.external_corpus(base, np.ones(n), labels, "EXT", "target", {}, set())

    def forbid_rng(*args, **kwargs):
        raise AssertionError("post hoc diagnostic consumed RNG")

    monkeypatch.setattr(np.random, "default_rng", forbid_rng)
    trace = {"policy": "task-thompson", "pool": "full", "seed": 3, "picks": np.arange(n),
             "screen_id": "target", "pmid": "EXT"}
    diagnostic = external.after_cycle3_diagnostics(corpus, 2, [trace], pub_id="EXT")[0]
    assert diagnostic["diagnostic_time"] == "after_cycle3" and diagnostic["acquired"] == 192
    assert diagnostic["effective_donor_count"] < 1.1
    assert diagnostic["donor_agreement_z"][0] > 0 > diagnostic["donor_agreement_z"][1]
    assert diagnostic["donor_weights"].sum() == pytest.approx(1)
    np.testing.assert_array_equal(diagnostic["picks"], np.arange(192))
    with pytest.raises(ValueError, match="another external target"):
        external.after_cycle3_diagnostics(
            corpus, 2, [{**trace, "screen_id": "different"}], pub_id="EXT")


def test_core_ineligible_pool_is_reported_without_fitting_or_synthetic_result():
    """The wrapper must preserve the frozen min-hits gate rather than invent a thin result."""
    corpus = append_target(base_corpus(n=5000), np.arange(5000) == 0)
    model = StubModel()
    result = external.run_one_target(
        corpus, 2, np.zeros((5000, 1)), np.zeros(5000), pub_id="EXT", model=model,
        embed_donors=lambda h: np.zeros((5000, 1)))
    assert result["status"] == "not_eligible"
    assert result["rows"] == result["traces"] == result["diagnostics"] == []
    assert result["n_hits"] == 1 and model.fits == 0
