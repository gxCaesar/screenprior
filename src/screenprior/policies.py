"""Acquisition policies, including the one this package exists for.

Every policy has the same signature and reads the same surrogate, so a difference between two rows
of a results table is a difference between the policies and not between their scaffolding.

THE IDEA. Uncertainty sampling, margin sampling, BADGE and coreset all reduce uncertainty about
INDIVIDUAL GENE LABELS. If a screen is approximately a point in a space spanned by screens that
have already been run, that is uncertainty about the wrong variable: what is worth resolving early
is WHICH SCREEN THIS IS, and what to acquire follows from the answer. `task_thompson` samples a
completed screen from a posterior over the corpus and takes that screen's best unacquired hit.

It has no tunable knob. The inverse temperature is fixed at 1.0 for the policy, for its ablation
and for its control alike, so no arm carries a degree of freedom another lacks.

THE PARAMETER IS AN INVERSE TEMPERATURE AND IS NAMED ONE. The weights are `exp(beta * z)`, so
raising it CONCENTRATES the distribution: on a five-donor example, beta from 0.25 to 8 takes the
effective number of donors from 4.63 to 1.00. Under the usual `softmax(z / T)` convention the same
sweep would run the other way, and an earlier version of this file called the parameter
`temperature` while implementing `exp(temperature * z)`. At the fixed value of 1.0 the two
conventions coincide exactly, which is why no reported number depended on the confusion and why it
survived: only the sweep panel and the Methods sentence read backwards.

TWO ARMS SHIP WITH IT AND ARE NOT OPTIONAL when the policy is reported:

  `uniform`   the removal ablation -- donors sampled with equal weight. Isolates "acquire from the
              corpus's hit sets at all" from "condition on the labels observed so far".
  `permuted`  the matched control -- the identical multiset of weights attached to the WRONG
              donors. Destroys the mechanism, preserves batch size, surrogate, seeds and fallback.
              Measured at -0.0034 against the removal where the policy itself gains +0.0121.
"""

from __future__ import annotations

import numpy as np
from sklearn.cluster import MiniBatchKMeans

from screenprior.discobax import DiscoBAX

__all__ = [
    "POLICIES",
    "badge",
    "badge_kmeanspp",
    "coreset",
    "donor_agreement",
    "donor_ranking_loss",
    "donor_rule",
    "farthest_first",
    "greedy",
    "kmeans_data",
    "margin_sample",
    "permute_within",
    "random_batch",
    "rgpe_donors",
    "size_popularity_strata",
    "soft_uncertain",
    "task_thompson",
    "top_uncertain",
]


def donor_agreement(donor_hits: np.ndarray, donor_tested: np.ndarray,
                    observed: np.ndarray) -> np.ndarray:
    """Hypergeometric z of each donor screen's hit set against the labels acquired so far.

    Restricted, per donor, to the acquired genes THAT DONOR TESTED. Without that restriction a
    donor with a boutique library is penalised for genes it never had the chance to call, and the
    posterior would track library size instead of phenotype.

    donor_hits, donor_tested: (n_acquired, n_donors) indicator matrices.
    observed: (n_acquired,) 0/1 labels.
    """
    if donor_hits.shape != donor_tested.shape:
        raise ValueError("donor hit and tested matrices must have the same shape")
    if len(observed) != donor_hits.shape[0]:
        raise ValueError("observed labels must have one entry per acquired gene")
    m = donor_tested.sum(axis=0)
    K = (donor_hits * donor_tested).sum(axis=0)
    k = (observed[:, None] * donor_tested).sum(axis=0)
    overlap = (observed[:, None] * donor_hits * donor_tested).sum(axis=0)
    expected = k * K / np.maximum(m, 1.0)
    variance = k * K * (m - K) * (m - k) / np.maximum(m * m * (m - 1.0), 1.0)
    return np.nan_to_num((overlap - expected) / np.sqrt(variance + 1e-9),
                         nan=0.0, posinf=0.0, neginf=0.0)


def farthest_first(X: np.ndarray, pool: np.ndarray, chosen: np.ndarray, batch: int) -> np.ndarray:
    """Greedy k-centre. Distances go through |a-b|^2 = |a|^2 - 2a.b + |b|^2 deliberately.

    The direct broadcast materialises pool x chosen x dim and reaches gigabytes on a genome-scale
    pool; this form never allocates more than pool x chosen.
    """
    Zp, Zc = X[pool], X[chosen]
    best = ((Zp ** 2).sum(axis=1)[:, None] + (Zc ** 2).sum(axis=1)[None, :]
            - 2.0 * (Zp @ Zc.T)).min(axis=1)
    picks = []
    for _ in range(batch):
        k = int(np.argmax(best))
        picks.append(pool[k])
        d = (Zp ** 2).sum(axis=1) + (Zp[k] ** 2).sum() - 2.0 * (Zp @ Zp[k])
        best = np.minimum(best, d)
        best[k] = -np.inf
    return np.array(picks)


def random_batch(rng, pool, batch, **_):
    return rng.choice(pool, size=batch, replace=False)


def greedy(rng, pool, batch, *, mean, **_):
    return pool[np.argsort(-mean[pool], kind="stable")[:batch]]


def top_uncertain(rng, pool, batch, *, spread, **_):
    return pool[np.argsort(-spread[pool], kind="stable")[:batch]]


def soft_uncertain(rng, pool, batch, *, spread, **_):
    # 1e-9, not a smaller epsilon: this value was used in the run that produced the reported
    # numbers, and the sampling probabilities depend on it. Neither choice is scientific;
    # reproducing the published run is the tiebreaker.
    weight = spread[pool] + 1e-9
    return rng.choice(pool, size=batch, replace=False, p=weight / weight.sum())


def coreset(rng, pool, batch, *, X, chosen, **_):
    return farthest_first(X, pool, chosen, batch)


def badge(rng, pool, batch, *, X, chosen, mean, **_):
    """Farthest-first in a gradient-scaled embedding, the geometry BADGE selects in."""
    scaled = X[pool] * (np.abs(mean[pool] - mean[pool].mean()) + 1e-9)[:, None]
    stacked = np.vstack([scaled, X[chosen]])
    take = farthest_first(stacked, np.arange(len(pool)),
                          np.arange(len(pool), len(pool) + len(chosen)), batch)
    return pool[take]


def margin_sample(rng, pool, batch, *, mean, **_):
    """Margin sampling for a binary label (Scheffer, Decomain and Wrobel 2001).

    The candidates whose predicted hit probability is closest to the decision boundary, ranked by
    |2p - 1| with p the surrogate mean clipped to [0, 1]. GeneDisco's version reads the spread of an
    ensemble's members; the shared surrogate exposes its mean and spread only, so this is the
    textbook binary form rather than a transcription of that one.
    """
    p = np.clip(mean[pool], 0.0, 1.0)
    return pool[np.argsort(np.abs(2.0 * p - 1.0), kind="stable")[:batch]]


def kmeans_data(rng, pool, batch, *, X, **_):
    """GeneDisco's k-means acquisition in the input space: cluster the pool into `batch` clusters
    and take the unacquired candidate nearest each centre.

    GeneDisco also ships a variant that clusters a network's learned embedding. A linear surrogate
    has no representation beyond its inputs, so for this surrogate the two coincide and only this
    one is run.
    """
    Z = X[pool]
    km = MiniBatchKMeans(n_clusters=batch, random_state=int(rng.integers(1 << 31)), n_init=3,
                         batch_size=1024).fit(Z)
    taken, out = set(), []
    for centre in km.cluster_centers_:
        for j in np.argsort(((Z - centre) ** 2).sum(axis=1), kind="stable"):
            if int(j) not in taken:
                taken.add(int(j))
                out.append(int(j))
                break
    for j in rng.permutation(len(Z)):                       # empty clusters can leave gaps
        if len(out) >= batch:
            break
        if int(j) not in taken:
            taken.add(int(j))
            out.append(int(j))
    return pool[np.array(out[:batch])]


def badge_kmeanspp(rng, pool, batch, *, X, mean, **_):
    """BADGE as Ash et al. (2020) define it, for a binary hit label read from the shared surrogate.

    BADGE embeds each candidate by the gradient of the loss at the label the model itself
    predicts, taken with respect to the output layer: (p - y_hat) times the penultimate features.
    A linear surrogate's penultimate features are its inputs, so the embedding here is
    (p - 1[p >= 0.5]) x, with p the surrogate mean clipped to [0, 1]. The batch is the k-means++
    seeding of those embeddings: the largest-norm candidate first, then each next one drawn with
    probability proportional to its squared distance from the nearest candidate already taken.

    `badge` above is the deterministic farthest-first variant every reported run used. A cold
    panel found that it is not the published algorithm, so this one ships beside it rather than
    replacing it, and the variant keeps its name only because the frozen tables carry it.
    """
    p = np.clip(mean[pool], 0.0, 1.0)
    g = (p - (p >= 0.5))[:, None] * X[pool]
    sq = (g ** 2).sum(axis=1)
    first = int(np.argmax(sq))
    picks = [first]
    d2 = np.maximum(sq + sq[first] - 2.0 * (g @ g[first]), 0.0)
    d2[first] = 0.0
    for _ in range(batch - 1):
        total = d2.sum()
        if total > 0:
            k = int(rng.choice(len(pool), p=d2 / total))
        else:
            # Every remaining candidate coincides with a centre already taken; k-means++ has no
            # preference among them, so draw uniformly from those not yet picked.
            k = int(rng.choice(np.setdiff1d(np.arange(len(pool)), picks)))
        picks.append(k)
        d2 = np.minimum(d2, np.maximum(sq + sq[k] - 2.0 * (g @ g[k]), 0.0))
        d2[picks] = 0.0
    return pool[np.array(picks)]


def size_popularity_strata(donor_hits: np.ndarray, donor_tested: np.ndarray,
                           bins: int = 5) -> np.ndarray:
    """Stratum of each donor by two properties that do not depend on the target's labels.

    Its number of hits, and the mean corpus hit frequency of those hits. A donor with many hits,
    or with hits among the genes most screens call, agrees with almost any target, so these are
    the target-independent structure a permutation of z across all donors destroys along with the
    target-specific part. Quantile bins of each, `bins` by `bins` strata.
    """
    called = donor_hits * donor_tested
    size = called.sum(axis=0).astype(float)
    frequency = called.sum(axis=1) / np.maximum(donor_tested.sum(axis=1), 1)
    popularity = (frequency[:, None] * called).sum(axis=0) / np.maximum(size, 1.0)

    def _bin(values):
        cuts = np.quantile(values, np.linspace(0, 1, bins + 1)[1:-1])
        return np.searchsorted(cuts, values, side="right")

    return _bin(size) * bins + _bin(popularity)


def permute_within(z: np.ndarray, strata: np.ndarray, rng) -> np.ndarray:
    """Shuffle z among donors of the same stratum only."""
    out = z.copy()
    for stratum in np.unique(strata):
        members = np.flatnonzero(strata == stratum)
        out[members] = z[rng.permutation(members)]
    return out


def donor_ranking_loss(donor_hits: np.ndarray, donor_tested: np.ndarray, observed: np.ndarray,
                       counts: np.ndarray) -> np.ndarray:
    """Each donor's pairwise ranking loss on the acquired labels, as RGPE scores a source model.

    A donor predicts hit or non-hit, so for a (hit, non-hit) pair of acquired genes it ranks the
    pair wrongly when it calls the non-hit and not the hit, and ties when it calls both or neither.
    Loss = (wrong + tied / 2) / pairs, over pairs of genes the donor tested; `counts` weights each
    acquired gene (a bootstrap resample). A donor with no testable pair scores 0.5, chance.
    """
    called = (donor_hits * donor_tested).astype(float)
    tested = donor_tested.astype(float)
    pos = counts * (observed > 0)
    neg = counts * (observed <= 0)
    pos_hit, pos_tested = pos @ called, pos @ tested
    neg_hit, neg_tested = neg @ called, neg @ tested
    pos_miss, neg_miss = pos_tested - pos_hit, neg_tested - neg_hit
    pairs = pos_tested * neg_tested
    loss = pos_miss * neg_hit + 0.5 * (pos_hit * neg_hit + pos_miss * neg_miss)
    return np.where(pairs > 0, loss / np.maximum(pairs, 1e-12), 0.5)


def rgpe_donors(rng, pool, batch, *, chosen, mean, donor_hits, donor_tested, observed,
                samples: int = 100, **_):
    """A ranking-weighted ensemble of donor screens, after RGPE (Feurer, Letham and Bakshy 2018).

    RGPE weights each source task by the probability that it ranks the target's observations best.
    Here the source models are the donors' hit sets and the uncertainty comes from bootstrapping the
    acquired labels: a donor's weight is the share of `samples` resamples in which its ranking loss
    is lowest, ties split evenly. Genes are then taken greedily by the weighted fraction of donors
    calling them, with the shared surrogate breaking ties and filling a short batch. It differs from
    task-Thompson in two places only: how donors are weighted, and argmax instead of sampling.
    Adapted rather than transcribed: RGPE also scores the target's own model, which is left out.
    """
    a = np.asarray(chosen)
    H, T, y = donor_hits[a].astype(float), donor_tested[a].astype(float), observed[a]
    wins = np.zeros(H.shape[1])
    for _ in range(samples):
        counts = np.bincount(rng.integers(0, len(a), size=len(a)), minlength=len(a)).astype(float)
        loss = donor_ranking_loss(H, T, y, counts)
        best = np.flatnonzero(loss == loss.min())
        wins[best] += 1.0 / len(best)
    w = wins / wins.sum()
    called = donor_hits[pool] * donor_tested[pool]
    score = (called * w).sum(axis=1) / np.maximum((donor_tested[pool] * w).sum(axis=1), 1e-12)
    order = np.lexsort((-mean[pool], -score))
    return pool[order[:batch]]


def donor_log_likelihood(donor_hits: np.ndarray, donor_tested: np.ndarray,
                         observed: np.ndarray) -> np.ndarray:
    """One-sided log-likelihood ratio of the acquired labels under each donor's two-rate model.

    The latent-bandit reading of the corpus: if the running screen is donor d, its hit rate among
    d's hits differs from its rate among d's non-hits. For each donor the acquired genes it tested
    are split that way, each part gets its own add-one-smoothed hit rate, and the log-likelihood is
    compared with one shared smoothed rate over the same genes. Donors whose hits are depleted for
    the observed hits score zero, so the posterior rewards resemblance and not anti-resemblance.

    donor_hits, donor_tested: (n_acquired, n_donors) indicator matrices; observed: (n_acquired,).
    """
    if donor_hits.shape != donor_tested.shape:
        raise ValueError("donor hit and tested matrices must have the same shape")
    if len(observed) != donor_hits.shape[0]:
        raise ValueError("observed labels must have one entry per acquired gene")
    T = donor_tested.astype(float)
    H = donor_hits.astype(float) * T
    y = np.asarray(observed, dtype=float)[:, None]
    n1, k1 = H.sum(axis=0), (y * H).sum(axis=0)
    n0, k0 = (T - H).sum(axis=0), (y * (T - H)).sum(axis=0)

    def loglik(k, n):
        p = (k + 1.0) / (n + 2.0)
        return k * np.log(p) + (n - k) * np.log1p(-p)

    llr = loglik(k1, n1) + loglik(k0, n0) - loglik(k1 + k0, n1 + n0)
    enriched = (k1 + 1.0) / (n1 + 2.0) > (k0 + 1.0) / (n0 + 2.0)
    return np.where(enriched, llr, 0.0)


def gene_thompson(rng, pool, batch, *, mean, spread, **_):
    """Batch Thompson sampling over genes on the shared surrogate, one draw per slot.

    Each slot draws a score for every pool gene from N(mean, spread^2), the bagged ridge's own mean
    and spread across bags, and takes the best gene not yet in the batch. It uses no completed
    screen, so beside task-Thompson it separates sampling per slot from sampling a completed screen.
    """
    mu, sd = mean[pool], spread[pool]
    taken = np.zeros(len(pool), dtype=bool)
    picks = []
    for _ in range(batch):
        draw = mu + sd * rng.standard_normal(len(pool))
        draw[taken] = -np.inf
        k = int(np.argmax(draw))
        taken[k] = True
        picks.append(k)
    return pool[np.array(picks)]


def task_thompson(rng, pool, batch, *, X, chosen, mean, donor_hits, donor_tested, observed,
                  beta=1.0, weights="posterior", **_):
    """Sample a completed screen from the posterior, take its best unacquired hit, repeat.

    weights: "posterior" for the policy, "uniform" for the removal ablation, "permuted" for the
    matched control, "permuted-within" for the stratified control, and "likelihood" for the
    latent-bandit comparator, which scores donors by `donor_log_likelihood` instead of the
    hypergeometric z. All share every other line, which is what makes them comparable.
    """
    if weights == "likelihood":
        z = donor_log_likelihood(donor_hits[chosen], donor_tested[chosen], observed[chosen])
    else:
        z = donor_agreement(donor_hits[chosen], donor_tested[chosen], observed[chosen])
    if weights == "uniform":
        z = np.zeros_like(z)
    elif weights == "permuted":
        z = rng.permutation(z)
    elif weights == "permuted-within":
        # The stricter control: the same shuffle, but only among donors alike in
        # hit-set size and frequent-hitter content, so target-independent structure survives.
        z = permute_within(z, size_popularity_strata(donor_hits, donor_tested), rng)
    elif weights not in ("posterior", "likelihood"):
        raise ValueError(f"unknown weights mode: {weights!r}")
    w = np.exp(beta * (z - z.max()))
    w /= w.sum()

    reachable = np.zeros(donor_hits.shape[0], dtype=bool)
    reachable[pool] = True
    fallback = pool[np.argsort(-mean[pool], kind="stable")]
    picked, seen = [], set()
    # Three draws per slot: a sampled donor may have nothing unacquired left to offer. Measured on
    # 153 screens, 99.8% of slots are filled by a donor and the fallback below is near-dead code --
    # which is what stops this policy from being greedy in disguise.
    for donor in rng.choice(len(w), size=batch * 3, replace=True, p=w):
        if len(picked) >= batch:
            break
        rows = np.flatnonzero(donor_hits[:, donor] & donor_tested[:, donor] & reachable)
        if not len(rows):
            continue
        for gene in rows[np.argsort(-mean[rows], kind="stable")]:
            if int(gene) not in seen:
                picked.append(int(gene))
                seen.add(int(gene))
                break
    for gene in fallback:
        if len(picked) >= batch:
            break
        if int(gene) not in seen:
            picked.append(int(gene))
            seen.add(int(gene))
    return np.array(picked[:batch])


def _thompson_with(weights):
    def policy(rng, pool, batch, **kwargs):
        return task_thompson(rng, pool, batch, **{**kwargs, "weights": weights})
    policy.__name__ = f"task_thompson_{weights}"
    policy.__doc__ = task_thompson.__doc__
    return policy


def donor_rule(rng, pool, batch, *, chosen, mean, donor_hits, donor_tested, observed,
               rule="sample", beta=1.0, k=5, prior=None, **_):
    """The same donor distribution, consumed four other ways. This is what earns the word Thompson.

    `task_thompson` samples one donor per slot. That is a choice, and until this existed the paper
    could not say what it bought: a method that samples from a distribution but wins for reasons
    unrelated to sampling is weighted retrieval with an expensive name. Four alternatives use the
    IDENTICAL z, the identical beta and the identical surrogate, and differ only in how the
    distribution is turned into a batch.

      sample  -- one draw per slot (what `task_thompson` does; kept here for a like-for-like arm)
      map     -- always the single highest-z donor; no randomness at all
      vote    -- the top-k donors by z, genes ranked by how many of the k call them a hit
      mean    -- rank genes by sum_d pi(d) * hit_d(g), the posterior-MEAN gene score
      batch   -- one donor drawn per cycle, filling all b slots from it

    `map` and `mean` are deterministic given the labels; `batch` samples but once per cycle rather
    than once per slot, which is the comparison that isolates within-batch donor diversity from
    sampling as such.

    THIS IS NOT A SECOND IMPLEMENTATION OF THE POLICY. `sample` reproduces `task_thompson` only up
    to the order in which it consumes the random stream, so the two are not bit-identical and the
    published arm must keep coming from `task_thompson`. The arm reported from here is `sample`,
    and it is reported only against the other four in this same function, never against the frozen
    tables.

    `prior` is pi_0 over donors, uniform when None. A non-uniform prior is what the publication-
    balanced variants need: two publications contribute 340 and 325 screens, so a uniform prior
    over screens is a prior that believes those two papers 665 times more than a paper with one.
    """
    z = donor_agreement(donor_hits[chosen], donor_tested[chosen], observed[chosen])
    w = np.exp(beta * (z - z.max()))
    if prior is not None:
        w = w * np.asarray(prior, dtype=np.float64)
    w /= w.sum()

    reachable = np.zeros(donor_hits.shape[0], dtype=bool)
    reachable[pool] = True
    offers = donor_hits & donor_tested & reachable[:, None]
    fallback = pool[np.argsort(-mean[pool], kind="stable")]

    def _fill(picked: list, seen: set) -> np.ndarray:
        for gene in fallback:
            if len(picked) >= batch:
                break
            if int(gene) not in seen:
                picked.append(int(gene))
                seen.add(int(gene))
        return np.array(picked[:batch])

    if rule in ("mean", "vote"):
        # A score over genes rather than a sequence of donors. `mean` weights every donor by its
        # posterior mass; `vote` gives the top k an equal voice and everyone else none, which is
        # the discrete version of the same idea and the one a practitioner would reach for.
        if rule == "mean":
            score = offers.astype(np.float64) @ w
        else:
            top = np.argsort(-z, kind="stable")[:k]
            score = offers[:, top].sum(axis=1).astype(np.float64)
        score = np.where(reachable, score, -np.inf)
        # Ties on the donor score are broken by the surrogate, not by gene index: at k=5 most of
        # the pool ties at zero, and breaking those by index would rank genes alphabetically.
        order = np.lexsort((-mean, -score))
        take = [int(g) for g in order if reachable[g]][:batch]
        return _fill(list(take), set(take))

    # batch repeats, not 3 * batch: a repeated donor yields its next-best gene each pass, so batch
    # passes is exactly enough to exhaust it, and the fallback handles a donor with fewer offers.
    if rule == "map":
        donors = np.repeat(int(np.argmax(z)), batch)
    elif rule == "batch":
        donors = np.repeat(int(rng.choice(len(w), p=w)), batch)
    elif rule == "sample":
        donors = rng.choice(len(w), size=batch * 3, replace=True, p=w)
    else:
        raise ValueError(f"unknown rule: {rule!r}")

    picked, seen = [], set()
    for donor in donors:
        if len(picked) >= batch:
            break
        rows = np.flatnonzero(offers[:, donor])
        for gene in rows[np.argsort(-mean[rows], kind="stable")]:
            if int(gene) not in seen:
                picked.append(int(gene))
                seen.add(int(gene))
                break
    return _fill(picked, seen)


def _rule_with(rule):
    def policy(rng, pool, batch, **kwargs):
        return donor_rule(rng, pool, batch, **{**kwargs, "rule": rule})
    policy.__name__ = f"donor_rule_{rule}"
    policy.__doc__ = donor_rule.__doc__
    return policy


# The removal and the control are FIRST-CLASS entries, not keyword arguments a caller has to
# remember. The README says they must be reported alongside the policy; a registry that cannot
# express them makes that instruction easy to skip.
POLICIES = {
    "random": random_batch,
    "greedy": greedy,
    "topuncertain": top_uncertain,
    "softuncertain": soft_uncertain,
    "coreset": coreset,
    "badge": badge,
    "badge/kmeans++": badge_kmeanspp,                        # the published algorithm
    "marginsample": margin_sample,
    "kmeans-data": kmeans_data,
    "rgpe-donors": rgpe_donors,                              # transfer-BO comparator
    "task-thompson": task_thompson,
    "task-thompson/uniform": _thompson_with("uniform"),      # removal ablation
    "task-thompson/permuted": _thompson_with("permuted"),    # matched control
    "task-thompson/permuted-within": _thompson_with("permuted-within"),  # stratified control
    "likelihood-donors": _thompson_with("likelihood"),       # latent-bandit likelihood posterior
    "gene-thompson": gene_thompson,                          # corpus-free batch Thompson
    # The ladder that establishes what SAMPLING buys, as registry entries rather than as keyword
    # arguments a caller has to remember -- the same reason the removal and the control are here.
    # `task-thompson/sample` is donor_rule's own like-for-like arm and is NOT the published one:
    # it consumes the random stream in a different order, so it is comparable to the other four
    # rules and to nothing else.
    "task-thompson/sample": _rule_with("sample"),
    "task-thompson/map": _rule_with("map"),                  # deterministic: the argmax donor
    "task-thompson/vote": _rule_with("vote"),                # top-k donors vote on genes
    "task-thompson/mean": _rule_with("mean"),                # posterior-mean gene score
    "task-thompson/batch": _rule_with("batch"),              # one donor per cycle, not per slot
    # DiscoBAX from the paper's equations on the shared surrogate; `screenprior.discobax` lists
    # every reading it had to choose. One instance, so the noise factor survives between calls.
    "discobax": DiscoBAX(),
}
