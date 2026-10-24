"""DiscoBAX as a comparator, written from the paper's equations.

Lyle, Mehrjou, Notin, Jesson, Bauer, Gal and Schwab, "DiscoBAX: Discovery of optimal intervention
sets in genomic experiment design", ICML 2023. Everything below is implemented from the paper's
mathematics; where the paper is silent, the value comes from the authors' released run script and
is labelled as configuration rather than as something derived.

THE SPLIT MATTERS AND IS NOT COSMETIC. The upstream repository states GPL-3.0 in its root LICENSE
and Apache-2.0 in its source headers -- the two are not compatible for this purpose, and no amount
of reading the repository resolves it. Algorithms and configuration values are not creative
expression; a transcription of someone's code is. So the algorithm comes from equations (1), (4),
(5), (6) and Algorithm 1, and the numbers come from `genedisco_experiment.sh` and
`apps/genedisco_experiment.py`, which are facts about how the published run was configured.

FROM THE PAPER
    (3)  additive noise, eta ~ GP(0, K_X), K_X an RBF kernel over the gene representation
    (4)  max_{S} E_eta [ max_{x in S} f_out(x; eta) ]
    (5)  f(S) = E_eta [ max_{x in S} max(0, f_out(x; eta)) ]      -- the clipped, submodular form
    Alg 1  greedy: S <- S + argmax_x E_eta [ max over S + {x} ]
    (1)  EIG(x) = H(f_ip(x) | D) - E_{p(S|D)} [ H(f_ip(x) | S, D) ]

TWO DEFECTS IN THE PAPER AS PRINTED, both resolved in favour of what makes the objective coherent,
and both recorded because a reader comparing this file to the PDF will otherwise think it is wrong:

  * Algorithm 1's box prints `E_eta[max_{y in S + {x}} f_out(x; eta)]` -- the maximum ranges over
    `y` while the expression names `x`. Read literally the inner maximum does nothing. Implemented
    as `max over y`, which is what equation (4) says and what makes the greedy step meaningful.
  * Algorithm 1's box omits the `max(0, .)` clipping that equation (5) introduces, and it is that
    clipping which makes the objective submodular (the paper's Observation 1). Implemented with
    the clipping, i.e. following equation (5) rather than the box.

WHERE THE PAPER IS SILENT, and these therefore come from the released configuration:
    subset size 20            `bax_subset_select_subset_size` (the paper gives only |S| = k)
    EIG samples 20            `bax_num_samples_EIG`
    entropy samples 20        `bax_num_samples_entropy`
    entropy estimator         the paper's equation (6) needs a density p(y | D, f); the released
                              code supplies a kernel estimator over the sampled predictions, and
                              the paper cites Gal & Ghahramani (2016) only for the samples
    noise outputscale 1.0     `bax_noise_outputscale` (the paper fixes the mean at zero and the
                              GeneDisco length scale at 1, and states no amplitude)

THE NOISE SAMPLE HAS NO USABLE LOW-RANK FORM. Measured on the real representation at the authors'
length scale: 56.8% of gene pairs sit above a kernel value of 1e-6 but the median pair is at
3.9e-5, so the spectrum is close to "identity plus local blocks". Random Fourier features at 2,048
dimensions miss a 0.05 tolerance at 0.0937 and Nystrom misses it at 1.0000 for every landmark count
tried. An exact Cholesky is therefore the only correct option -- which is why the authors' code
requires a GPU, though the arithmetic does not: 15,300 genes is 0.94 GB and 1.19 TFLOP.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "DISCOBAX_CONFIG",
    "GaussianFieldNoise",
    "expected_information_gain",
    "predictive_entropy",
    "rbf_kernel",
    "subset_select",
]

# Values the paper does not state, from the authors' released GeneDisco configuration.
DISCOBAX_CONFIG = {"subset_size": 20, "num_samples_eig": 20, "num_samples_entropy": 20,
                   "noise_lengthscale": 1.0, "noise_outputscale": 1.0, "noise_draws": 20}


def rbf_kernel(A: np.ndarray, B: np.ndarray, lengthscale: float = 1.0,
               outputscale: float = 1.0) -> np.ndarray:
    """k(a, b) = outputscale * exp(-||a - b||^2 / (2 * lengthscale^2)).

    Expanded rather than broadcast: the direct form materialises |A| x |B| x d and reaches
    gigabytes on a genome-scale pool.
    """
    squared = ((A ** 2).sum(axis=1)[:, None] + (B ** 2).sum(axis=1)[None, :] - 2.0 * (A @ B.T))
    return outputscale * np.exp(-np.maximum(squared, 0.0) / (2.0 * lengthscale ** 2))


class GaussianFieldNoise:
    """Draws from eta ~ GP(0, K_X), equation (3), by an exact Cholesky.

    Exact because no approximation of this kernel survives a check: see the module docstring.
    """

    def __init__(self, X: np.ndarray, lengthscale: float = 1.0, outputscale: float = 1.0,
                 jitter: float = 1e-6):
        K = rbf_kernel(np.ascontiguousarray(X, dtype=np.float32),
                       np.ascontiguousarray(X, dtype=np.float32),
                       lengthscale, outputscale).astype(np.float32)
        K[np.diag_indices_from(K)] += jitter
        self.root = np.linalg.cholesky(K)

    def draw(self, rng: np.random.Generator) -> np.ndarray:
        return self.root @ rng.normal(size=self.root.shape[1]).astype(np.float32)


def subset_select(values: np.ndarray, subset_size: int) -> list:
    """Algorithm 1 over pre-drawn noisy readouts, maximising equation (5).

    `values` is (n_draws, n_candidates), each row one realisation of max(0, f_out(., eta)); the
    expectation over eta is the mean across rows.

    The greedy step adds the candidate maximising E_eta[max over S + {x}]. Because the running
    maximum over S is fixed within a draw, that equals mean_r max(running[r], values[r, x]) and one
    broadcast replaces the loop over candidates without changing the selection.
    """
    if values.ndim != 2:
        raise ValueError(f"values must be (draws, candidates), got shape {values.shape}")
    if not 1 <= subset_size <= values.shape[1]:
        raise ValueError(f"subset_size {subset_size} outside 1..{values.shape[1]}")
    first = int(np.argmax(values.mean(axis=0)))     # S is empty, so the max over S + {x} is x
    chosen, running = [first], values[:, first].copy()
    for _ in range(subset_size - 1):
        gain = np.maximum(running[:, None], values).mean(axis=0)
        gain[chosen] = -np.inf
        pick = int(np.argmax(gain))
        chosen.append(pick)
        running = np.maximum(running, values[:, pick])
    return chosen


def predictive_entropy(samples: np.ndarray, precision: float = 1.0) -> np.ndarray:
    """Equation (6): entropy of the predictive distribution from M model samples.

    The paper writes H = E[log p] without the leading minus sign that entropy carries, and does not
    say how p is formed from the samples. Implemented as the negative mean log of a Gaussian kernel
    estimator over the samples, which is the form the released code uses and the only reading under
    which the quantity is an entropy.

    `samples` is (n_points, M). Returns (n_points,).
    """
    if samples.ndim != 2:
        raise ValueError(f"samples must be (points, draws), got shape {samples.shape}")
    n_draws = samples.shape[1]
    difference = samples[:, :, None] - samples[:, None, :]
    block = -precision / 2.0 * difference ** 2
    peak = block.max(axis=2, keepdims=True)
    log_density = (peak + np.log(np.exp(block - peak).sum(axis=2, keepdims=True)))[:, :, 0]
    mean_log = log_density.sum(axis=1) / n_draws
    return -(mean_log - np.log(n_draws) - 0.5 * np.log(2 * np.pi) + 0.5 * np.log(precision))


def expected_information_gain(prior_samples: np.ndarray, posterior_samples: np.ndarray
                              ) -> np.ndarray:
    """Equation (1): H(f(x) | D) minus the mean of H(f(x) | S_j, D) over sampled algorithm outputs.

    `prior_samples` is (n_candidates, M). `posterior_samples` is (n_sets, n_candidates, M), one
    block per S_j drawn from p(S | D).
    """
    if posterior_samples.ndim != 3:
        raise ValueError("posterior_samples must be (sets, candidates, draws)")
    before = predictive_entropy(prior_samples)
    after = np.stack([predictive_entropy(block) for block in posterior_samples])
    return before - after.mean(axis=0)
