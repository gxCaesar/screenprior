"""Hand-computed checks on the DiscoBAX arm, and the invariants every policy in the harness owes.

The equations themselves are checked in `test_comparators.py`. These check the wiring: that one
call of the arm composes them as Algorithm 2 says, on numbers small enough to do by hand, and that
the arm behaves as a policy must inside the loop.
"""

from __future__ import annotations

import numpy as np
import pytest

from screenprior import discobax
from screenprior.comparators import GaussianFieldNoise, expected_information_gain
from screenprior.corpus import Corpus
from screenprior.discobax import (
    DiscoBAX,
    conditioned_samples,
    noise_block,
    rank_by_information_gain,
    sampled_subset,
)
from screenprior.harness import RunConfig, run_screen
from screenprior.policies import POLICIES

PUBLISHED = ("random", "greedy", "topuncertain", "softuncertain", "coreset", "badge",
             "task-thompson", "task-thompson/permuted", "task-thompson/uniform")


def small_arm() -> DiscoBAX:
    return DiscoBAX(subset_size=4, num_samples_eig=4, num_samples_entropy=4, noise_draws=4)


def screen(seed: int = 0, n: int = 120, d: int = 3) -> tuple:
    """Features in which the label genuinely depends on the first coordinate, 30 genes acquired."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    observed = (X[:, 0] + 0.5 * rng.normal(size=n) > 1.0).astype(float)
    return {"X": X, "chosen": np.arange(30), "observed": observed}, np.arange(30, n)


def test_the_subset_of_a_three_gene_set_matches_a_hand_computation():
    """Algorithm 1 under additive noise, three genes and two noise draws.

    Sample f = (0.5, 0.45, 0). Draw 0: eta = (0.5, 0.45, -1); draw 1: eta = (-1, -1, 0.6).
    f + eta clipped at zero: draw 0 gives (1.0, 0.9, 0), draw 1 gives (0, 0, 0.6).
    Means 0.5, 0.45, 0.3, so gene 0 is taken first.
    Adding gene 1: mean of (max(1.0, 0.9), max(0, 0)) = 0.5. Adding gene 2: mean of (1.0, 0.6) = 0.8.
    Gene 2 comes second although gene 1 has the larger value on its own, which is what equation (5)
    is for: a gene that pays off on the draws the first one misses.
    """
    sample = np.array([0.5, 0.45, 0.0])
    eta = np.array([[0.5, -1.0], [0.45, -1.0], [-1.0, 0.6]])          # (genes, draws)
    assert sampled_subset(sample, eta, 2) == [0, 2]


def test_the_subset_clips_at_zero_as_equation_5_does():
    """f = 0 for two genes. Gene 0 draws eta = (3, -5), gene 1 draws (1, 1).
    Clipped, gene 0 scores (3 + 0) / 2 = 1.5 and gene 1 scores 1.0, so gene 0 is taken. Without the
    clipping gene 0 would score (3 - 5) / 2 = -1 and gene 1 would be taken instead."""
    eta = np.array([[3.0, -5.0], [1.0, 1.0]])
    assert sampled_subset(np.zeros(2), eta, 1) == [0]


def test_information_gain_and_the_batch_on_three_genes_by_hand():
    """Equation (6) at M = 2 and equation (1) over two sampled sets.

    Two samples d apart have, under a unit Gaussian kernel, the entropy
        h(d) = 0.5 log(2 pi) + log 2 - log(1 + exp(-d^2 / 2)),
    so h(0) = 0.918939, h(1) = 1.138009, h(2) = 1.485158 and h(3) = 1.601038.
    Prior sample gaps 2, 1, 0; gaps given set 1: 0, 1, 0; given set 2: 1, 1, 3. Then
        EIG(gene 0) = h(2) - (h(0) + h(1)) / 2 = 0.456684
        EIG(gene 1) = h(1) - (h(1) + h(1)) / 2 = 0
        EIG(gene 2) = h(0) - (h(0) + h(3)) / 2 = -0.341050
    and a batch of two takes genes 0 and 1, reported in pool indices.
    """
    prior = np.array([[0.0, 2.0], [0.0, 1.0], [0.0, 0.0]])
    posterior = np.stack([np.array([[0.0, 0.0], [0.0, 1.0], [0.0, 0.0]]),
                          np.array([[0.0, 1.0], [0.0, 1.0], [0.0, 3.0]])])
    gain = expected_information_gain(prior, posterior)
    assert gain == pytest.approx([0.456684, 0.0, -0.341050], abs=1e-6)
    pool = np.array([10, 11, 12])
    assert rank_by_information_gain(pool, 2, prior, posterior).tolist() == [10, 11]
    assert rank_by_information_gain(pool, 1, prior, posterior).tolist() == [10]


class _Recorder:
    """Stands in for the surrogate and records what the conditioning step asks it to fit."""

    def __init__(self):
        self.calls = []

    def bags(self, X, y, train, alpha, n, seed):
        self.calls.append((X.copy(), y.copy(), np.asarray(train).copy(), alpha, n, seed))
        return np.tile(np.arange(X.shape[0], dtype=float)[:, None], (1, n))


def test_conditioning_adds_each_subset_member_once_with_its_sampled_value():
    """The pseudo-observations are the subset's rows of X, labelled with the sample's values there,
    appended to the acquired genes; an acquired member (gene 0) is added again, not skipped, and
    only the real candidates' predictions come back."""
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([1.0, 0.0, 0.0, 0.0])
    sample = np.array([0.9, 0.2, 0.7, 0.1])
    recorder = _Recorder()
    out = conditioned_samples(recorder, X, y, np.array([0, 1]), [2, 0], sample, 10.0, 3, 5)
    X_aug, y_aug, train, alpha, draws, seed = recorder.calls[0]
    assert X_aug[4:, 0].tolist() == [2.0, 0.0]
    assert y_aug.tolist() == [1.0, 0.0, 0.0, 0.0, 0.7, 0.9]
    assert train.tolist() == [0, 1, 4, 5]
    assert (alpha, draws, seed) == (10.0, 3, 5)
    assert out.shape == (4, 3)
    assert out[:, 0].tolist() == [0.0, 1.0, 2.0, 3.0]


def test_the_noise_block_is_the_field_drawn_once_per_column():
    """Drawing all realisations in one product must give what successive `draw` calls give."""
    field = GaussianFieldNoise(np.random.default_rng(0).normal(size=(40, 3)))
    stream = np.random.default_rng(9)
    one_at_a_time = np.column_stack([field.draw(stream) for _ in range(4)])
    together = noise_block(field.root, np.random.default_rng(9), 4)
    assert together.shape == (40, 4)
    assert np.allclose(together, one_at_a_time, atol=1e-5)


def test_the_arm_is_deterministic_given_its_stream_and_moves_with_it():
    """Same stream, same batch, including from a fresh instance with no cached factor; a different
    stream must be able to change the batch, or the arm is not sampling anything."""
    kwargs, pool = screen()
    first = small_arm()(np.random.default_rng(4), pool, 8, **kwargs)
    again = small_arm()(np.random.default_rng(4), pool, 8, **kwargs)
    assert first.tolist() == again.tolist()
    arm = small_arm()
    seen = {tuple(sorted(arm(np.random.default_rng(s), pool, 8, **kwargs).tolist()))
            for s in range(6)}
    assert len(seen) > 1


def test_the_arm_never_reads_a_label_it_has_not_acquired():
    kwargs, pool = screen()
    flipped = kwargs["observed"].copy()
    flipped[pool] = 1.0 - flipped[pool]
    honest = small_arm()(np.random.default_rng(2), pool, 8, **kwargs)
    leaked = small_arm()(np.random.default_rng(2), pool, 8, **{**kwargs, "observed": flipped})
    assert honest.tolist() == leaked.tolist()


def test_the_noise_field_is_factorised_once_per_feature_matrix(monkeypatch):
    """The Cholesky is the expensive step; a target's seeds and cycles share one feature matrix."""
    built = []
    real = discobax.GaussianFieldNoise

    def counting(X, *args, **kwargs):
        built.append(X.shape)
        return real(X, *args, **kwargs)

    monkeypatch.setattr(discobax, "GaussianFieldNoise", counting)
    arm = small_arm()
    kwargs, pool = screen()
    arm(np.random.default_rng(0), pool, 4, **kwargs)
    arm(np.random.default_rng(1), pool[4:], 4, **kwargs)
    assert len(built) == 1
    other, pool = screen(seed=1)
    arm(np.random.default_rng(0), pool, 4, **other)
    assert len(built) == 2


def synthetic_corpus(n_genes: int = 300, n_screens: int = 8, seed: int = 0) -> tuple:
    """Screens that share hit programmes, so the donor arms have something to read.

    The features are continuous and spread out. The integer programme column alone puts many genes
    on top of one another, and the RBF kernel over them is then too close to singular for the
    float32 Cholesky the noise field uses."""
    rng = np.random.default_rng(seed)
    programme = rng.integers(0, 3, size=n_genes)
    screen_programme = rng.integers(0, 3, size=n_screens)
    hit = np.zeros((n_genes, n_screens), dtype=bool)
    for s in range(n_screens):
        member = programme == screen_programme[s]
        hit[member, s] = rng.random(member.sum()) < 0.35
        hit[~member, s] = rng.random((~member).sum()) < 0.02
    corpus = Corpus(genes=np.array([f"G{i}" for i in range(n_genes)]),
                    screen_id=np.array([str(100 + s) for s in range(n_screens)]),
                    pmid=np.array([f"p{s}" for s in range(n_screens)]),
                    cell_line=np.array(["K562"] * n_screens),
                    phenotype=np.array(["x"] * n_screens),
                    hit=hit, tested=np.ones_like(hit))
    features = np.column_stack([1.5 * rng.normal(size=(n_genes, 6)),
                                programme + 0.3 * rng.normal(size=n_genes)])
    return corpus, features, rng.normal(size=n_genes)


def test_in_the_loop_the_arm_never_acquires_a_gene_twice_and_leaves_other_streams_alone():
    """Appended after the nine published arms, the arm must not move a single pick of theirs, and
    its own acquisitions must be distinct and exactly the budget."""
    corpus, features, effect = synthetic_corpus()
    config = RunConfig(batch=12, cycles=3, seeds=(0, 1), pool="full", min_pool=100, min_hits=5,
                       record_picks=True)
    registry = {**POLICIES, "discobax": small_arm()}
    base_rows, base_traces = run_screen(corpus, 0, features, effect, policies=PUBLISHED,
                                        config=config, registry=registry)
    rows, traces = run_screen(corpus, 0, features, effect, policies=PUBLISHED + ("discobax",),
                              config=config, registry=registry)
    ours = [t["picks"] for t in traces if t["policy"] == "discobax"]
    assert len(ours) == len(config.seeds)
    for picks in ours:
        assert len(picks) == len(set(picks.tolist())) == config.batch * config.cycles
    kept = [t for t in traces if t["policy"] != "discobax"]
    assert [(t["seed"], t["policy"]) for t in kept] == [(t["seed"], t["policy"])
                                                        for t in base_traces]
    for before, after in zip(base_traces, kept):
        assert before["picks"].tolist() == after["picks"].tolist()
    assert [r for r in rows if r["policy"] != "discobax"] == base_rows


def test_the_registered_arm_runs_at_the_released_configuration():
    arm = POLICIES["discobax"]
    assert isinstance(arm, DiscoBAX)
    assert (arm.subset_size, arm.num_samples_eig, arm.num_samples_entropy, arm.noise_draws,
            arm.lengthscale, arm.outputscale) == (20, 20, 20, 20, 1.0, 1.0)
