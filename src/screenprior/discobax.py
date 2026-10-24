"""DiscoBAX as one more policy in the shared harness, wired from `screenprior.comparators`.

Lyle, Mehrjou, Notin, Jesson, Bauer, Gal and Schwab, ICML 2023, Algorithm 2 with Algorithm 1 inside
it. Each cycle draws l samples of the intermediate phenotype f_ip from the posterior, runs
SubsetSelect on each under the additive noise of equation (3), and acquires the candidates whose
prediction the sampled subsets would sharpen most, by the expected information gain of equation (1)
with the entropy estimate of equation (6). The equations, the noise sampler, the greedy step and the
entropy estimator are `comparators`, written from the paper. This module connects them to the
harness, which is where every reading the paper leaves open has to be fixed. Each one is listed
below as a choice, so that it can be checked against the paper rather than against another
implementation. No line here was written from the authors' code; the sizes in `DISCOBAX_CONFIG` are
the released run's configuration, as `comparators` records.

WHAT THE HARNESS FIXES. The arm reads the surrogate every other policy reads, a bootstrap-bagged
ridge refit on the acquired labels only, on the unstandardised feature matrix the harness passes to
every policy. DiscoBAX's own network is not used. The paper draws its samples of f_ip by MC dropout
(equation 6); here each sample is one bootstrap fit, from `BaggedRidge.bags`.

THE CHOICES, where the paper or `comparators` leaves the reading open:

 1. Ridge penalty. The harness gives a policy the bagged ridge's mean and spread but neither the
    alpha behind them nor the run seed. The arm re-tunes alpha with the surrogate's own `tune` (same
    grid, folds and criterion) on the acquired labels, seeding the fold shuffle from its own stream.
    Where two grid values score alike it can pick a different one from the alpha behind `mean`.
 2. The set SubsetSelect searches. Algorithm 2 runs it over the finite set X. Here X is every gene
    the screen can acquire (tested by the target and inside the non-essential pool), acquired genes
    included; only the batch is restricted to genes not yet acquired.
 3. Values at acquired genes. SubsetSelect sees the sample's own predictions there, not the observed
    labels, because Algorithm 2 hands it the sampled f_ip as drawn.
 4. Noise. eta ~ GP(0, K), K the RBF kernel over the same gene set at length scale 1 and amplitude 1,
    factorised once per feature matrix by an exact Cholesky (`GaussianFieldNoise`: float32, jitter
    1e-6). Each SubsetSelect call gets its own `noise_draws` fresh realisations, since Algorithm 1
    takes the noise distribution as an input and estimates its own expectation over it.
 5. Conditioning on S_j. The paper estimates H(f_ip(x) | S, D) "by conditioning a posterior on the
    values of elements in the set S". Here the bags are refit on D plus one pseudo-observation
    (s, f_ip,j(s)) for every s in S_j, already-acquired members included.
 6. Entropy samples. H(f_ip(x) | D) and each H(f_ip(x) | S_j, D) are estimated from their own
    `num_samples_entropy` bootstrap fits, separate from the `num_samples_eig` fits that SubsetSelect
    runs on: equation (6)'s M samples and Algorithm 2's l samples are different symbols.
 7. Entropy estimator. `comparators.predictive_entropy`, a Gaussian kernel at precision 1 over the
    samples. Log densities are averaged arithmetically over samples (equation 6) and entropies
    arithmetically over the sampled sets (equation 1).
 8. The batch. The `batch` unacquired genes with the largest expected information gain, ties
    broken by pool position with a stable sort ("selecting the B points with the highest EIG").
 9. f_ip is the binary hit label the surrogate regresses, as it is for every arm. The paper's f_ip
    is a continuous readout.
10. Sizes. Subset size, both sample counts and the noise draws come from `DISCOBAX_CONFIG`. A gene
    set smaller than the subset size, which only a test produces, caps the subset at the set.

Labels of genes not yet acquired are zeroed before any fit, so the arm cannot read them.
"""

from __future__ import annotations

import hashlib

import numpy as np

from screenprior.comparators import (
    DISCOBAX_CONFIG,
    GaussianFieldNoise,
    expected_information_gain,
    subset_select,
)
from screenprior.surrogate import BaggedRidge

__all__ = [
    "DiscoBAX",
    "conditioned_samples",
    "noise_block",
    "rank_by_information_gain",
    "sampled_subset",
]


def _seed(rng: np.random.Generator) -> int:
    return int(rng.integers(1 << 31))


def noise_block(root: np.ndarray, rng: np.random.Generator, draws: int) -> np.ndarray:
    """`draws` realisations of eta, one per column, (n, draws).

    Column r is what the r-th successive `GaussianFieldNoise.draw(rng)` would return: the normals
    are taken from the stream in the same order and multiplied by the Cholesky root in one product
    instead of one at a time.
    """
    normals = rng.normal(size=(draws, root.shape[1])).astype(np.float32)
    return root @ normals.T


def sampled_subset(sample: np.ndarray, eta: np.ndarray, subset_size: int) -> list:
    """Algorithm 1 on one posterior sample under additive noise.

    f_out(x; eta) = f_ip(x) + eta(x), equation (3), clipped at zero as equation (5) requires.
    `sample` is (n,) and `eta` is (n, draws); `subset_select` averages over the draws.
    """
    values = np.maximum(0.0, sample[None, :] + eta.T)
    return subset_select(values, subset_size)


def conditioned_samples(model, X: np.ndarray, y: np.ndarray, chosen: np.ndarray, subset,
                        sample: np.ndarray, alpha: float, draws: int, seed: int) -> np.ndarray:
    """Samples of f_ip given D and given that the algorithm returned `subset`, (n, draws).

    Conditioning is by pseudo-observation: each member s of the subset joins the training set once
    more, with the value `sample` (the posterior draw that produced the subset) gives it, and the
    bags are refit on the union. Predictions are returned for the n real candidates only.
    """
    n = X.shape[0]
    subset = np.asarray(subset, dtype=int)
    X_aug = np.vstack([X, X[subset]])
    y_aug = np.concatenate([y, sample[subset]])
    train = np.concatenate([np.asarray(chosen, dtype=int), n + np.arange(len(subset))])
    return model.bags(X_aug, y_aug, train, alpha, draws, seed)[:n]


def rank_by_information_gain(pool: np.ndarray, batch: int, prior: np.ndarray,
                             posterior: np.ndarray) -> np.ndarray:
    """The `batch` members of `pool` with the largest expected information gain, equation (1).

    `prior` is (len(pool), M), samples given D; `posterior` is (sets, len(pool), M), samples given
    D and each sampled set. The sort is stable, so equal gains keep pool order.
    """
    gain = expected_information_gain(prior, posterior)
    return pool[np.argsort(-gain, kind="stable")[:batch]]


class DiscoBAX:
    """The policy. Called like every entry of `POLICIES`; keeps the noise factor between calls.

    The Cholesky root depends only on the feature matrix, which the harness holds fixed across a
    target's seeds and cycles, so it is built once per matrix and reused. It is keyed by the
    matrix's bytes, not its identity: a freed array's id can be reused by the next screen's matrix,
    and the cache would then hand back the wrong field.
    """

    def __init__(self, subset_size: int = DISCOBAX_CONFIG["subset_size"],
                 num_samples_eig: int = DISCOBAX_CONFIG["num_samples_eig"],
                 num_samples_entropy: int = DISCOBAX_CONFIG["num_samples_entropy"],
                 noise_draws: int = DISCOBAX_CONFIG["noise_draws"],
                 lengthscale: float = DISCOBAX_CONFIG["noise_lengthscale"],
                 outputscale: float = DISCOBAX_CONFIG["noise_outputscale"],
                 model: BaggedRidge = None):
        self.subset_size = subset_size
        self.num_samples_eig = num_samples_eig
        self.num_samples_entropy = num_samples_entropy
        self.noise_draws = noise_draws
        self.lengthscale = lengthscale
        self.outputscale = outputscale
        self.model = BaggedRidge() if model is None else model
        self._key = None
        self._field = None

    def field(self, X: np.ndarray) -> GaussianFieldNoise:
        data = np.ascontiguousarray(X)
        key = (data.shape, data.dtype.str,
               hashlib.blake2b(data.tobytes(), digest_size=16).hexdigest())
        if key != self._key:
            self._key, self._field = None, None          # release the old root before building
            self._field = GaussianFieldNoise(data, self.lengthscale, self.outputscale)
            self._key = key
        return self._field

    def __call__(self, rng, pool, batch, *, X, chosen, observed, **_):
        n = X.shape[0]
        chosen = np.asarray(chosen)
        y = np.zeros(n)
        y[chosen] = observed[chosen]
        model, sets, m = self.model, self.num_samples_eig, self.num_samples_entropy

        alpha = model.tune(X[chosen], y[chosen], _seed(rng))
        prior = model.bags(X, y, chosen, alpha, m, _seed(rng))
        samples = model.bags(X, y, chosen, alpha, sets, _seed(rng))
        eta = noise_block(self.field(X).root, rng, sets * self.noise_draws)
        size = min(self.subset_size, n)

        posterior = np.empty((sets, len(pool), m))
        for j in range(sets):
            block = eta[:, j * self.noise_draws:(j + 1) * self.noise_draws]
            subset = sampled_subset(samples[:, j], block, size)
            posterior[j] = conditioned_samples(model, X, y, chosen, subset, samples[:, j], alpha,
                                               m, _seed(rng))[pool]
        return rank_by_information_gain(pool, batch, prior[pool], posterior)
