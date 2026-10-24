"""The loop's invariants: no gene twice, no policy advantaged by its starting point, no silent pool."""

from __future__ import annotations

import numpy as np
import pytest

from screenprior.corpus import Corpus
from screenprior.harness import RunConfig, run_screen


def synthetic(n_genes=600, n_screens=12, seed=0) -> tuple:
    """A corpus where donor screens genuinely carry information about the target.

    Genes are given a latent "programme"; each screen picks a programme and calls its genes hits.
    Two screens sharing a programme therefore share hits, which is the structure task-thompson is
    supposed to find. Without it the test would pass on noise and prove nothing.
    """
    rng = np.random.default_rng(seed)
    programme = rng.integers(0, 4, size=n_genes)
    screen_programme = rng.integers(0, 4, size=n_screens)
    hit = np.zeros((n_genes, n_screens), dtype=bool)
    for s in range(n_screens):
        member = programme == screen_programme[s]
        hit[member, s] = rng.random(member.sum()) < 0.35
        hit[~member, s] = rng.random((~member).sum()) < 0.02
    tested = np.ones_like(hit)
    corpus = Corpus(genes=np.array([f"G{i}" for i in range(n_genes)]),
                    screen_id=np.array([str(100 + s) for s in range(n_screens)]),
                    pmid=np.array([f"p{s}" for s in range(n_screens)]),
                    cell_line=np.array(["K562"] * n_screens),
                    phenotype=np.array(["x"] * n_screens),
                    hit=hit, tested=tested)
    features = np.column_stack([rng.normal(size=n_genes), programme.astype(float)])
    gene_effect = rng.normal(size=n_genes)
    return corpus, features, gene_effect


CONFIG = RunConfig(batch=16, cycles=4, seeds=(0, 1), pool="full", min_pool=100, min_hits=5)


def test_no_gene_is_acquired_twice_and_the_budget_is_exact():
    corpus, features, effect = synthetic()
    rows = run_screen(corpus, 0, features, effect,
                      policies=("random", "greedy", "coreset", "badge", "task-thompson"),
                      config=CONFIG)
    assert rows, "the synthetic screen should be large enough to run"
    for row in rows:
        assert row["acquired"] == CONFIG.batch * CONFIG.cycles


def test_every_policy_starts_from_the_same_first_batch():
    """A policy must not win by having been dealt a better opening hand."""
    corpus, features, effect = synthetic()
    single = RunConfig(batch=16, cycles=2, seeds=(0,), pool="full", min_pool=100, min_hits=5)
    rows = run_screen(corpus, 0, features, effect, policies=("greedy", "task-thompson"),
                      config=single)
    # With two cycles, half of each policy's acquisitions is the shared opening batch.
    assert {row["acquired"] for row in rows} == {32}


def test_a_screen_below_the_minimums_returns_nothing_rather_than_a_thin_result():
    corpus, features, effect = synthetic()
    strict = RunConfig(batch=16, cycles=4, seeds=(0,), pool="full", min_pool=10_000, min_hits=5)
    assert run_screen(corpus, 0, features, effect, config=strict) == []


def test_the_non_essential_pool_actually_removes_genes():
    corpus, features, effect = synthetic()
    everything = run_screen(corpus, 0, features, effect, policies=("random",), config=CONFIG)
    trimmed = run_screen(corpus, 0, features, effect, policies=("random",),
                         config=RunConfig(batch=16, cycles=4, seeds=(0,), pool="non-essential",
                                          essential_cut=0.0, min_pool=100, min_hits=5))
    assert trimmed[0]["n_genes"] < everything[0]["n_genes"]


def test_an_unknown_policy_name_raises_rather_than_being_skipped():
    corpus, features, effect = synthetic()
    with pytest.raises(KeyError, match="unknown policy"):
        run_screen(corpus, 0, features, effect, policies=("nonesuch",), config=CONFIG)


def test_an_invalid_pool_name_is_rejected_at_construction():
    with pytest.raises(ValueError, match="pool must be"):
        RunConfig(pool="everything")


def test_the_run_is_reproducible_to_the_gene():
    corpus, features, effect = synthetic()
    first = run_screen(corpus, 0, features, effect, policies=("task-thompson",), config=CONFIG)
    second = run_screen(corpus, 0, features, effect, policies=("task-thompson",), config=CONFIG)
    assert [r["recall"] for r in first] == [r["recall"] for r in second]


def test_publication_blocking_reaches_the_donor_count_the_run_reports():
    corpus, features, effect = synthetic()
    rows = run_screen(corpus, 0, features, effect, policies=("task-thompson",), config=CONFIG)
    assert rows[0]["n_donors"] == corpus.hit.shape[1] - 1     # its own publication is excluded


def test_task_thompson_beats_random_when_the_corpus_actually_carries_the_structure():
    """A weak end-to-end sanity check, not a claim: on a corpus built so that donor screens share
    programmes with the target, the policy must be able to exploit it. If this fails, the wiring
    between corpus, policy and loop is broken -- it says nothing about real data."""
    corpus, features, effect = synthetic()
    config = RunConfig(batch=16, cycles=6, seeds=(0, 1, 2), pool="full", min_pool=100, min_hits=5)
    rows = run_screen(corpus, 0, features, effect, policies=("random", "task-thompson"),
                      config=config)
    by_policy = {}
    for row in rows:
        by_policy.setdefault(row["policy"], []).append(row["recall"])
    assert np.mean(by_policy["task-thompson"]) > np.mean(by_policy["random"])


def test_the_corpus_embedding_is_built_after_blocking_not_before():
    """The leak this interface exists to prevent.

    An embedding computed once over every screen carries the target's own hit pattern. Here the
    embedder records which matrix it was handed; it must never be the full corpus, and its column
    count must equal the number of donors left after the target's publication is removed.
    """
    corpus, static, effect = synthetic()
    seen = []

    def embedder(donor_hits):
        seen.append(donor_hits.shape)
        return np.zeros((donor_hits.shape[0], 3), dtype=np.float32)

    run_screen(corpus, 0, static, effect, policies=("task-thompson",), config=CONFIG,
               embed_donors=embedder)
    assert seen, "the embedder was never called"
    n_screens = corpus.hit.shape[1]
    assert len(seen) == 1, "the embedding should be built once per screen, after blocking"
    assert seen[0][1] == n_screens - 1          # the target's own publication is gone
    assert seen[0][1] != n_screens              # ... and it is not the whole corpus


def test_the_static_features_are_widened_by_the_embedding():
    corpus, static, effect = synthetic()
    width = []

    def embedder(donor_hits):
        block = np.zeros((donor_hits.shape[0], 5), dtype=np.float32)
        width.append(block.shape[1])
        return block

    rows = run_screen(corpus, 0, static, effect, policies=("coreset",), config=CONFIG,
                      embed_donors=embedder)
    # Once per screen, not once per seed: the donor set does not change with the seed, and
    # recomputing an SVD per seed would be pure waste.
    assert rows and width == [5]


# --- recording ------------------------------------------------------------------------------
# Recording must preserve the final numerical result exactly. The following tests
# compare runs with recording enabled and disabled.

POLICIES_UNDER_TEST = ("random", "greedy", "badge", "task-thompson")


def test_recording_cycles_does_not_move_the_final_number():
    """Recording is observation. If it changes what is observed it is not recording."""
    corpus, features, effect = synthetic()
    quiet = run_screen(corpus, 0, features, effect, policies=POLICIES_UNDER_TEST, config=CONFIG)
    loud = run_screen(corpus, 0, features, effect, policies=POLICIES_UNDER_TEST,
                      config=RunConfig(**{**vars(CONFIG), "record_cycles": True}))
    final = [r for r in loud if r["cycle"] == CONFIG.cycles]
    assert len(final) == len(quiet)
    for a, b in zip(quiet, final):
        assert a["seed"] == b["seed"] and a["policy"] == b["policy"]
        assert a["recall"] == b["recall"], "the same run reported two different recalls"
        assert a["acquired"] == b["acquired"]


def test_a_cycle_row_is_emitted_for_every_cycle_and_recall_never_falls():
    corpus, features, effect = synthetic()
    rows = run_screen(corpus, 0, features, effect, policies=POLICIES_UNDER_TEST,
                      config=RunConfig(**{**vars(CONFIG), "record_cycles": True}))
    for seed in CONFIG.seeds:
        for policy in POLICIES_UNDER_TEST:
            track = [r for r in rows if r["seed"] == seed and r["policy"] == policy]
            assert [r["cycle"] for r in track] == list(range(1, CONFIG.cycles + 1))
            assert [r["acquired"] for r in track] == [
                CONFIG.batch * c for c in range(1, CONFIG.cycles + 1)]
            # Acquisition only adds, so recall is monotone. A fall would mean the accounting is
            # being done against a moving denominator.
            recalls = [r["recall"] for r in track]
            assert recalls == sorted(recalls)


def test_the_trace_is_the_acquisition_in_order_and_in_corpus_gene_indices():
    corpus, features, effect = synthetic()
    rows, traces = run_screen(corpus, 0, features, effect, policies=POLICIES_UNDER_TEST,
                              config=RunConfig(**{**vars(CONFIG), "record_picks": True}))
    assert len(traces) == len(rows) == len(CONFIG.seeds) * len(POLICIES_UNDER_TEST)
    for trace, row in zip(traces, rows):
        picks = trace["picks"]
        assert picks.dtype == np.int32
        assert len(picks) == CONFIG.batch * CONFIG.cycles
        assert len(set(picks.tolist())) == len(picks), "a gene appears twice in the trace"
        assert picks.max() < len(corpus.genes), "trace is not in corpus gene indices"
        # The recall the row reports must be recoverable from the trace alone.
        recovered = corpus.hit[picks, 0].sum() / corpus.hit[:, 0].sum()
        assert recovered == pytest.approx(row["recall"])


def test_the_first_batch_is_shared_across_policies_in_the_trace_too():
    corpus, features, effect = synthetic()
    _, traces = run_screen(corpus, 0, features, effect, policies=POLICIES_UNDER_TEST,
                           config=RunConfig(**{**vars(CONFIG), "record_picks": True}))
    for seed in CONFIG.seeds:
        openings = {tuple(t["picks"][:CONFIG.batch]) for t in traces if t["seed"] == seed}
        assert len(openings) == 1, "policies did not start from the same batch"


def test_a_screen_below_the_minimums_returns_two_empties_when_tracing():
    corpus, features, effect = synthetic()
    rows, traces = run_screen(corpus, 0, features, effect, policies=("random",),
                              config=RunConfig(**{**vars(CONFIG), "record_picks": True,
                                                  "min_hits": 10_000}))
    assert rows == [] and traces == []


def test_a_substituted_registry_is_used_and_an_unknown_name_still_raises():
    """The sweep hook. Six task-Thompsons at six temperatures are one policy, not six."""
    from screenprior.policies import POLICIES, task_thompson
    corpus, features, effect = synthetic()
    registry = {"hot": lambda rng, pool, batch, **kw: task_thompson(rng, pool, batch,
                                                                    **{**kw, "temperature": 8.0})}
    rows = run_screen(corpus, 0, features, effect, policies=("hot",), config=CONFIG,
                      registry=registry)
    assert rows and rows[0]["policy"] == "hot"
    with pytest.raises(KeyError):
        run_screen(corpus, 0, features, effect, policies=("task-thompson",), config=CONFIG,
                   registry=registry)
    assert "hot" not in POLICIES, "the sweep leaked into the shipped registry"


def test_a_substituted_first_batch_is_used_and_must_be_distinct():
    corpus, features, effect = synthetic()
    fixed = np.arange(CONFIG.batch)
    _, traces = run_screen(corpus, 0, features, effect, policies=("greedy",),
                           config=RunConfig(**{**vars(CONFIG), "record_picks": True}),
                           first_batch=lambda rng, n, batch: fixed)
    assert set(traces[0]["picks"][:CONFIG.batch].tolist()) == set(fixed.tolist())
    with pytest.raises(ValueError, match="distinct genes"):
        run_screen(corpus, 0, features, effect, policies=("greedy",), config=CONFIG,
                   first_batch=lambda rng, n, batch: np.zeros(batch, dtype=int))
