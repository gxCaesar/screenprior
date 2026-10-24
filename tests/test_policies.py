"""Known-answer checks on the parts of the acquisition code that decide a reported number."""

from __future__ import annotations

import numpy as np
import pytest

from screenprior import policies
from screenprior.policies import (
    POLICIES,
    badge,
    coreset,
    donor_agreement,
    donor_log_likelihood,
    farthest_first,
    gene_thompson,
    greedy,
    task_thompson,
    top_uncertain,
)


def test_donor_agreement_matches_a_hand_computed_hypergeometric_z():
    """Two donors, four acquired genes, arithmetic done by hand.

    Donor 0 tested all four and called {0, 1} hits; the acquired hits are exactly {0, 1}.
    m = 4, K = 2, k = 2, overlap = 2.  expected = kK/m = 1.
    variance = kK(m-K)(m-k) / (m^2 (m-1)) = 2*2*2*2 / (16*3) = 1/3.
    z = (2 - 1) / sqrt(1/3) = sqrt(3).
    Donor 1 called {2, 3}, disjoint from the acquired hits, so overlap = 0 and z = -sqrt(3).
    """
    donor_hits = np.array([[1, 0], [1, 0], [0, 1], [0, 1]], dtype=float)
    donor_tested = np.ones((4, 2), dtype=float)
    observed = np.array([1.0, 1.0, 0.0, 0.0])
    z = donor_agreement(donor_hits, donor_tested, observed)
    assert z[0] == pytest.approx(np.sqrt(3.0))
    assert z[1] == pytest.approx(-np.sqrt(3.0))


def test_donor_agreement_ignores_genes_a_donor_never_tested():
    """A donor is scored only on the acquired genes inside its own library.

    Without this, a small-library donor is punished for genes it could not have called, and the
    posterior tracks library size rather than phenotype.
    """
    donor_hits = np.array([[1.0], [1.0], [0.0], [0.0]])
    tested_all = np.ones((4, 1))
    tested_half = np.array([[1.0], [1.0], [0.0], [0.0]])   # only saw the first two genes
    observed = np.array([1.0, 1.0, 0.0, 0.0])
    full = donor_agreement(donor_hits, tested_all, observed)
    partial = donor_agreement(donor_hits, tested_half, observed)
    assert full[0] != pytest.approx(partial[0])
    assert np.isfinite(partial).all()


def test_donor_agreement_is_zero_when_nothing_has_been_observed_as_a_hit():
    donor_hits = np.array([[1.0, 0.0], [0.0, 1.0]])
    donor_tested = np.ones((2, 2))
    z = donor_agreement(donor_hits, donor_tested, np.zeros(2))
    assert np.allclose(z, 0.0)


def test_farthest_first_matches_the_naive_broadcast_it_replaces():
    """The squared-norm identity is an optimisation; it must not change which genes are selected."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 8))
    pool, chosen = np.arange(0, 150), np.arange(150, 200)

    def naive(batch):
        picks, current = [], list(chosen)
        for _ in range(batch):
            d = np.linalg.norm(X[pool][:, None, :] - X[current][None, :, :], axis=2).min(axis=1)
            d[[np.where(pool == p)[0][0] for p in picks]] = -np.inf
            k = int(np.argmax(d))
            picks.append(pool[k])
            current.append(pool[k])
        return picks

    assert list(farthest_first(X, pool, chosen, 6)) == naive(6)


def test_task_thompson_removal_and_control_change_only_the_weights():
    """All three modes must return a full, duplicate-free batch drawn from the pool."""
    rng_state = 20260811
    n, n_donors, batch = 400, 30, 16
    rng = np.random.default_rng(rng_state)
    donor_hits = (rng.random((n, n_donors)) < 0.08)
    donor_tested = np.ones((n, n_donors), dtype=bool)
    observed = (rng.random(n) < 0.08).astype(float)
    mean = rng.normal(size=n)
    chosen = np.arange(0, 64)
    pool = np.arange(64, n)

    for mode in ("posterior", "uniform", "permuted"):
        picked = task_thompson(np.random.default_rng(rng_state), pool, batch, X=None,
                               chosen=chosen, mean=mean, donor_hits=donor_hits,
                               donor_tested=donor_tested, observed=observed, weights=mode)
        assert len(picked) == batch
        assert len(set(picked.tolist())) == batch
        assert set(picked.tolist()) <= set(pool.tolist())


def test_task_thompson_rejects_an_unknown_weights_mode():
    with pytest.raises(ValueError, match="unknown weights mode"):
        task_thompson(np.random.default_rng(0), np.arange(10), 2, X=None,
                      chosen=np.arange(5), mean=np.zeros(10),
                      donor_hits=np.ones((10, 2), dtype=bool),
                      donor_tested=np.ones((10, 2), dtype=bool),
                      observed=np.ones(10), weights="best")


def test_task_thompson_falls_back_when_the_corpus_offers_nothing():
    """With no donor hits at all, every slot must come from the surrogate and the batch still fills.

    This is the degenerate case the fallback exists for. It is exercised here because on real data
    it almost never fires -- 99.8% of slots are donor-filled -- so nothing else would reach it.
    """
    n, batch = 100, 8
    mean = np.arange(n, dtype=float)
    picked = task_thompson(np.random.default_rng(0), np.arange(20, n), batch, X=None,
                           chosen=np.arange(20), mean=mean,
                           donor_hits=np.zeros((n, 3), dtype=bool),
                           donor_tested=np.ones((n, 3), dtype=bool),
                           observed=np.zeros(n))
    assert sorted(picked.tolist()) == sorted(range(n - batch, n))   # exactly the top of `mean`


@pytest.mark.parametrize("name", sorted(POLICIES))
def test_every_policy_returns_a_clean_batch_from_the_pool(name):
    rng_state = 7
    n, batch = 300, 12
    rng = np.random.default_rng(rng_state)
    X = rng.normal(size=(n, 6))
    mean, spread = rng.random(n), rng.random(n) + 0.01
    chosen, pool = np.arange(40), np.arange(40, n)
    kwargs = {"X": X, "chosen": chosen, "mean": mean, "spread": spread,
              "donor_hits": (rng.random((n, 12)) < 0.1),
              "donor_tested": np.ones((n, 12), dtype=bool),
              "observed": (rng.random(n) < 0.1).astype(float)}
    picked = POLICIES[name](np.random.default_rng(rng_state), pool, batch, **kwargs)
    assert len(picked) == batch
    assert len(set(np.asarray(picked).tolist())) == batch
    assert set(np.asarray(picked).tolist()) <= set(pool.tolist())


@pytest.mark.parametrize("policy", [greedy, top_uncertain, coreset, badge])
def test_deterministic_policies_repeat_exactly(policy):
    rng_state = 3
    n, batch = 200, 10
    rng = np.random.default_rng(rng_state)
    X = rng.normal(size=(n, 5))
    mean, spread = rng.random(n), rng.random(n) + 0.01
    chosen, pool = np.arange(30), np.arange(30, n)
    kwargs = {"X": X, "chosen": chosen, "mean": mean, "spread": spread}
    first = policy(np.random.default_rng(rng_state), pool, batch, **kwargs)
    second = policy(np.random.default_rng(rng_state), pool, batch, **kwargs)
    assert list(first) == list(second)


def test_the_removal_and_the_control_are_registry_entries_not_forgettable_kwargs():
    """They must be reportable the same way any other policy is, or reporting them is optional."""
    assert {"task-thompson", "task-thompson/uniform", "task-thompson/permuted"} <= set(POLICIES)
    rng_state, n, batch = 11, 300, 12
    rng = np.random.default_rng(rng_state)
    kwargs = {"X": None, "chosen": np.arange(40), "mean": rng.normal(size=n),
              "donor_hits": (rng.random((n, 9)) < 0.1),
              "donor_tested": np.ones((n, 9), dtype=bool),
              "observed": (rng.random(n) < 0.1).astype(float)}
    pool = np.arange(40, n)
    direct = task_thompson(np.random.default_rng(rng_state), pool, batch, weights="uniform",
                           **kwargs)
    viaregistry = POLICIES["task-thompson/uniform"](np.random.default_rng(rng_state), pool, batch,
                                                    **kwargs)
    assert list(direct) == list(viaregistry)


def test_beta_concentrates_the_donor_distribution_rather_than_flattening_it():
    """The parameter is an INVERSE temperature, and this pins which way it runs.

    The weights are `exp(beta * z)`. Raising beta concentrates; under the usual `softmax(z / T)`
    reading it would flatten. An earlier version named the parameter `temperature` while
    implementing this, so the Methods sentence and the sweep panel described the opposite of what
    the code did. At the fixed value of 1.0 the two conventions coincide exactly, so no reported
    number ever depended on it and nothing in the outputs could have caught it. A test can.
    """
    genes, donors = 40, 5
    hits = np.zeros((genes, donors), dtype=bool)
    for d in range(donors):
        hits[: 20 - 3 * d, d] = True          # donor 0 agrees most, donor 4 least
    tested = np.ones_like(hits)
    observed = np.zeros(genes)
    observed[:20] = 1.0

    z = donor_agreement(hits, tested, observed)
    effective = []
    for beta in (0.25, 1.0, 4.0, 8.0):
        w = np.exp(beta * (z - z.max()))
        w /= w.sum()
        effective.append(1.0 / np.square(w).sum())

    assert effective == sorted(effective, reverse=True), (
        f"effective donor count {effective} does not fall as beta rises; the parameter is an "
        "inverse temperature and the implementation must concentrate, not flatten")
    assert effective[0] > effective[-1] + 0.5, (
        "the sweep barely moves the distribution, so this test would not catch a sign flip")


def _ladder_fixture():
    """A pool where donors genuinely disagree AND no single donor owns the posterior.

    The first version of this made one donor a perfect match, which put essentially all posterior
    mass on it -- so `batch` drew that donor every time and looked deterministic. That was the
    fixture being degenerate, not the policy being wrong, and it is exactly the condition under
    which this whole ladder measures nothing. The effective donor count is asserted below so the
    fixture cannot quietly collapse again.
    """
    rng = np.random.default_rng(7)
    genes, donors = 300, 20
    hits = np.zeros((genes, donors), dtype=bool)
    truth = rng.choice(genes, size=60, replace=False)
    for d in range(donors):
        # Graded overlap with the truth: donor d shares 10 + d hits with it, so the z values form a
        # ladder of their own and several donors stay plausible.
        share = truth[: 10 + d]
        rest = rng.choice(np.setdiff1d(np.arange(genes), truth), size=40, replace=False)
        hits[np.concatenate([share, rest]), d] = True
    tested = np.ones_like(hits)
    observed = np.zeros(genes)
    observed[truth] = 1.0
    return ({"chosen": np.arange(64), "mean": rng.random(genes), "donor_hits": hits,
             "donor_tested": tested, "observed": observed}, np.arange(64, genes))


def test_the_ladder_fixture_has_a_spread_posterior():
    """Guards the two tests below: on a collapsed posterior they would pass for the wrong reason."""
    kwargs, _ = _ladder_fixture()
    c = kwargs["chosen"]
    z = donor_agreement(kwargs["donor_hits"][c], kwargs["donor_tested"][c], kwargs["observed"][c])
    w = np.exp(z - z.max())
    w /= w.sum()
    effective = 1.0 / np.square(w).sum()
    assert effective > 3.0, (
        f"only {effective:.2f} donors carry the posterior; a fixture this concentrated makes "
        "`batch` look deterministic and the ladder look like it measures nothing")


def test_the_deterministic_rungs_are_deterministic_and_the_sampling_ones_are_not():
    """The ladder only isolates sampling if its rungs differ in exactly that respect.

    A ladder whose 'deterministic' arms quietly consume randomness, or whose 'sampling' arms
    happen to be deterministic on this data, measures nothing -- and either would be invisible in
    a results table, which would show five plausible numbers.
    """
    kwargs, pool = _ladder_fixture()
    for rule in ("map", "mean", "vote"):
        a = policies.donor_rule(np.random.default_rng(1), pool, 32, rule=rule, **kwargs)
        b = policies.donor_rule(np.random.default_rng(999), pool, 32, rule=rule, **kwargs)
        assert np.array_equal(a, b), f"{rule} is supposed to be deterministic and moved with the seed"
    for rule in ("sample", "batch"):
        seen = {tuple(policies.donor_rule(np.random.default_rng(s), pool, 32, rule=rule, **kwargs))
                for s in range(8)}
        assert len(seen) > 1, f"{rule} is supposed to sample and did not move across eight seeds"


def test_one_donor_per_batch_really_uses_one_donor():
    """`batch` is the control that separates within-batch donor diversity from sampling as such.

    If it silently fell back to the surrogate for most of its slots it would be greedy wearing the
    label, and the comparison against `sample` would be measuring the fallback instead of the
    difference the arm is named for.
    """
    kwargs, pool = _ladder_fixture()
    hits = kwargs["donor_hits"]
    for seed in range(6):
        picked = policies.donor_rule(np.random.default_rng(seed), pool, 32, rule="batch", **kwargs)
        # Every gene a donor supplied is a hit in that donor, so the supplying donor's column
        # covers them all; at most one donor can do that unless two donors coincide.
        covering = [d for d in range(hits.shape[1]) if hits[picked, d].sum() >= 1]
        from_donor = max(int(hits[picked, d].sum()) for d in covering)
        assert from_donor >= 20, (
            f"seed {seed}: the best single donor supplied only {from_donor} of 32 genes, so this "
            "arm is filling most of its batch from the fallback rather than from one donor")


def test_badge_kmeanspp_takes_the_largest_gradient_embedding_first():
    """The published BADGE seeds k-means++ at the largest (p - y_hat) x; the variant did not.

    Hand-computed. Surrogate means 0.9, 0.4, 0.6, 0.5 give y_hat 1, 0, 1, 1 and p - y_hat -0.1,
    +0.4, -0.4, -0.5. With x = (1,0), (0,1), (2,0), (0,0.5) the embeddings are (-0.1,0), (0,0.4),
    (-0.8,0), (0,-0.25), squared norms 0.01, 0.16, 0.64, 0.0625: candidate 2 comes first.
    """
    from screenprior.policies import badge_kmeanspp
    X = np.array([[1.0, 0.0], [0.0, 1.0], [2.0, 0.0], [0.0, 0.5]])
    mean = np.array([0.9, 0.4, 0.6, 0.5])
    pool = np.arange(4)
    assert badge_kmeanspp(np.random.default_rng(0), pool, 1, X=X, mean=mean).tolist() == [2]


def test_badge_kmeanspp_draws_the_second_point_by_squared_distance():
    """After candidate 2, squared distances to (-0.8,0) are 0.49, 0.80, 0, 0.7025 (sum 1.9925), so
    candidate 1 should come second with probability 0.80 / 1.9925 = 0.4015 and candidate 2 never."""
    from screenprior.policies import badge_kmeanspp
    X = np.array([[1.0, 0.0], [0.0, 1.0], [2.0, 0.0], [0.0, 0.5]])
    mean = np.array([0.9, 0.4, 0.6, 0.5])
    pool = np.arange(4)
    rng = np.random.default_rng(0)
    second = [badge_kmeanspp(rng, pool, 2, X=X, mean=mean)[1] for _ in range(20_000)]
    assert 2 not in second
    assert np.mean(np.array(second) == 1) == pytest.approx(0.80 / 1.9925, abs=0.012)


def test_badge_kmeanspp_never_repeats_a_candidate_even_when_embeddings_coincide():
    """Five identical embeddings leave every squared distance at zero after the first pick; the
    batch must still hold five distinct candidates rather than the first one five times."""
    from screenprior.policies import badge_kmeanspp
    X = np.ones((5, 3))
    mean = np.full(5, 0.2)
    picks = badge_kmeanspp(np.random.default_rng(1), np.arange(5), 5, X=X, mean=mean)
    assert sorted(picks.tolist()) == [0, 1, 2, 3, 4]


def test_stratified_permutation_never_moves_a_weight_across_strata():
    """The permuted control destroyed target-independent donor structure along with the mechanism;
    the stratified one must keep every z inside its own stratum and only reorder it there."""
    from screenprior.policies import permute_within
    z = np.array([1.0, 2.0, 3.0, 10.0, 20.0, 30.0])
    strata = np.array([0, 0, 0, 1, 1, 1])
    out = permute_within(z, strata, np.random.default_rng(3))
    assert sorted(out[:3].tolist()) == [1.0, 2.0, 3.0]
    assert sorted(out[3:].tolist()) == [10.0, 20.0, 30.0]


def test_size_popularity_strata_by_hand():
    """Four donors over three genes. Donor hit sets {g0}, {g1}, {g0,g2}, {g0,g1}: counts 1, 1, 2, 2.
    Gene hit frequencies 3/4, 2/4, 1/4, so the mean frequency of each donor's hits is 0.75, 0.5,
    (0.75+0.25)/2 = 0.5 and (0.75+0.5)/2 = 0.625. Two bins: sizes cut at 1.5 give 0, 0, 1, 1;
    popularity cut at its median 0.5625 gives 1, 0, 0, 1. Strata = size * 2 + popularity."""
    from screenprior.policies import size_popularity_strata
    hits = np.array([[1, 0, 1, 1],
                     [0, 1, 0, 1],
                     [0, 0, 1, 0]])
    tested = np.ones_like(hits)
    assert size_popularity_strata(hits, tested, bins=2).tolist() == [1, 0, 2, 3]


def test_donor_ranking_loss_by_hand():
    """Two hits and two non-hits acquired, one donor calling the first hit and the first non-hit.
    Pairs: (h1,n1) tie, (h1,n2) right, (h2,n1) wrong, (h2,n2) tie -> (1 + 2 * 0.5) / 4 = 0.5.
    A second donor calling exactly the two hits ranks all four pairs right: loss 0."""
    from screenprior.policies import donor_ranking_loss
    observed = np.array([1, 1, 0, 0])
    hits = np.array([[1, 1], [0, 1], [1, 0], [0, 0]])
    tested = np.ones_like(hits)
    loss = donor_ranking_loss(hits, tested, observed, np.ones(4))
    assert loss.tolist() == [0.5, 0.0]


def test_rgpe_donors_puts_its_weight_on_the_donor_that_ranks_the_labels_best():
    """With the perfect donor of the test above, every resample that holds a hit and a non-hit
    makes it the unique best, so the genes it calls among the pool come first."""
    from screenprior.policies import rgpe_donors
    hits = np.array([[1, 1], [0, 1], [1, 0], [0, 0], [0, 1], [1, 0]])
    tested = np.ones_like(hits)
    observed = np.array([1, 1, 0, 0, 0, 0])
    mean = np.zeros(6)
    picks = rgpe_donors(np.random.default_rng(0), np.array([4, 5]), 1, chosen=np.arange(4),
                        mean=mean, donor_hits=hits, donor_tested=tested, observed=observed)
    assert picks.tolist() == [4]


def test_margin_sample_takes_the_predictions_nearest_one_half():
    """Means 0.1, 0.45, 0.9, 0.55 give |2p - 1| = 0.8, 0.1, 0.8, 0.1: candidates 1 and 3, in
    pool order on the tie because the sort is stable."""
    from screenprior.policies import margin_sample
    mean = np.array([0.1, 0.45, 0.9, 0.55])
    assert margin_sample(None, np.arange(4), 2, mean=mean).tolist() == [1, 3]


def test_kmeans_data_takes_one_candidate_from_each_well_separated_cluster():
    """Two tight clusters far apart and a batch of two: one pick must come from each."""
    from screenprior.policies import kmeans_data
    X = np.vstack([np.zeros((10, 2)) + [0, 0], np.zeros((10, 2)) + [100, 100]])
    X += np.random.default_rng(0).normal(scale=0.01, size=X.shape)
    picks = kmeans_data(np.random.default_rng(1), np.arange(20), 2, X=X)
    assert sorted(int(p) // 10 for p in picks) == [0, 1]


def test_donor_log_likelihood_rewards_resemblance_and_not_anti_resemblance():
    """A likelihood posterior over donors must not reward a donor whose hits are the target's
    non-hits; a two-sided ratio would, and the comparator would then copy the wrong screen."""
    observed = np.array([1.0, 1.0, 0.0, 0.0])
    hits = np.array([[1, 0, 1], [1, 0, 0], [0, 1, 0], [0, 1, 0]], dtype=bool)
    tested = np.array([[1, 1, 1], [1, 1, 0], [1, 1, 1], [1, 1, 0]], dtype=bool)
    llr = donor_log_likelihood(hits, tested, observed)
    # A: hit rates 3/4 among its hits and 1/4 among its non-hits, against 1/2 shared, so
    # 4 ln(3/4) - 4 ln(1/2) = 4 ln 1.5. B is the mirror image and scores zero. C tested two of the
    # four genes: 2 ln(2/3) - 2 ln(1/2) = 2 ln(4/3).
    assert llr == pytest.approx([4 * np.log(1.5), 0.0, 2 * np.log(4 / 3)])


def test_gene_thompson_with_no_spread_is_greedy_and_never_repeats():
    """With zero spread every draw is the mean, so the batch must be the greedy top-k in order; with
    a large spread it must still fill the batch with distinct genes."""
    pool = np.array([0, 1, 2, 3])
    mean = np.array([0.1, 0.9, 0.5, 0.3])
    got = gene_thompson(np.random.default_rng(0), pool, 2, mean=mean, spread=np.zeros(4))
    assert list(got) == [1, 2]
    wide = gene_thompson(np.random.default_rng(1), pool, 4, mean=mean, spread=np.full(4, 50.0))
    assert sorted(wide) == [0, 1, 2, 3]
