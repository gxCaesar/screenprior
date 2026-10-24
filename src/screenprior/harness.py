"""The sequential loop, with the choices that decide a number made in one place.

Everything here is a decision that has changed a result at least once in this project, so each is a
named argument rather than a constant buried in a script:

  `block_cell_line`  publication blocking is mandatory and lives in `Corpus`; this is the sharper
                     control on top of it
  `pool`             `"non-essential"` removes pan-essential genes from what can be ACQUIRED, not
                     merely from what is scored. The published protocol restricts neither, and
                     spends 24-43% of its budget on genes its own endpoint counts as zero
  `seed`             every policy in a run shares the first batch and then draws from its own
                     stream, so two policies differ in their choices and not in their luck

THE ORDER OF `policies` IS PART OF THE EXPERIMENT. Each policy is seeded `seed * 31 + 101 * slot`
where `slot` is its POSITION in the argument, so passing the same policies in a different order
gives every one of them a different stream and a different set of acquisitions. Measured on three
screens, reordering moved per-policy mean recall by up to 2.6e-02, which is larger than the margin
this project reports. The alternative -- seeding from the name -- was rejected because Python
randomises string hashing per process, which is worse: it makes a single run irreproducible while
looking deterministic. The order below fixes the stream associated with each arm and is
part of what a reader has to match to reproduce a table:

    random, greedy, topuncertain, softuncertain, coreset, badge,
    task-thompson, task-thompson/permuted, task-thompson/uniform

The loop returns one row per (screen, seed, policy) and never aggregates. Aggregation is a
reporting decision and it belongs to whoever is reporting.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from screenprior.policies import POLICIES
from screenprior.surrogate import BaggedRidge

__all__ = ["RunConfig", "run_screen"]


@dataclass
class RunConfig:
    batch: int = 64
    cycles: int = 10
    seeds: tuple = (0, 1, 2)
    pool: str = "non-essential"
    block_cell_line: bool = False
    essential_cut: float = -0.5
    min_pool: int = 5_000
    min_hits: int = 30
    model: BaggedRidge = field(default_factory=BaggedRidge)
    # Recording is off by default and the default path must stay byte-identical, because the
    # published numbers came from it. What these buy is everything the loop used to discard: a
    # recall per cycle rather than only at the end, and WHICH genes were acquired. Without the
    # second, no question about what the method actually selects can be asked at all.
    record_cycles: bool = False
    record_picks: bool = False

    def __post_init__(self) -> None:
        if self.pool not in ("non-essential", "full"):
            raise ValueError(f"pool must be 'non-essential' or 'full', not {self.pool!r}")
        if self.batch < 1 or self.cycles < 2:
            raise ValueError("need at least one gene per batch and two cycles")


def run_screen(corpus, target: int, features: np.ndarray, gene_effect: np.ndarray,
               policies=("random", "badge", "task-thompson"),
               config: RunConfig = None, embed_donors=None, registry=None,
               first_batch=None) -> list:
    """Run every policy on one target screen and return one row per (seed, policy).

    `features` is the STATIC half of the representation, (n_genes, d) -- DepMap co-essentiality,
    which does not depend on the corpus. `gene_effect` is the DepMap mean gene effect used only to
    define the non-essential pool; it is not a label and no policy sees it.

    `registry` substitutes the policy table, for sweeps over a policy's own parameters -- a
    temperature sweep needs six task-Thompsons, and putting them in POLICIES would ship six
    policies that are one policy. `first_batch` substitutes the opening draw, for asking where the
    advantage begins rather than assuming it begins at cycle two. Both default to the published
    behaviour and are the only way to vary these without editing the package.

    `embed_donors` is a callable (donor_hit_matrix) -> (n_genes, k), applied AFTER blocking and
    concatenated to `features`. It is a callable rather than an array because the donor set changes
    with the target: an embedding computed once over the whole corpus would carry the target
    screen's own hit pattern into the representation that ranks it, which is answer-key leakage
    arriving through a factorisation. `screenprior.features.corpus_embedder` returns one.
    """
    config = config or RunConfig()
    table = POLICIES if registry is None else registry
    donors = corpus.donors_for(target, block_cell_line=config.block_cell_line)
    donor_hits = corpus.hit[:, donors]
    donor_tested = corpus.tested[:, donors]
    if embed_donors is not None:
        features = np.hstack([features, embed_donors(donor_hits)])

    keep = corpus.tested[:, target].copy()
    if config.pool == "non-essential":
        keep &= gene_effect >= config.essential_cut
    idx = np.flatnonzero(keep)
    labels = corpus.hit[idx, target].astype(np.float64)
    if len(idx) < config.min_pool or labels.sum() < config.min_hits:
        return ([], []) if config.record_picks else []

    # float64 throughout: ridge on float32 features gives different low-order digits,
    # and the run that produced the reported numbers was float64.
    X = np.asarray(features[idx], dtype=np.float64)
    H, T = donor_hits[idx], donor_tested[idx]
    n, total = len(idx), float(labels.sum())
    rows, traces = [], []

    for seed in config.seeds:
        # One shared opening batch per seed: policies must differ in what they choose next, not in
        # where they started.
        first = (np.random.default_rng(seed).choice(n, size=config.batch, replace=False)
                 if first_batch is None
                 else np.asarray(first_batch(np.random.default_rng(seed), n, config.batch)))
        if len(set(first.tolist())) != config.batch:
            raise ValueError(f"the opening batch holds {len(set(first.tolist()))} distinct genes, "
                             f"not {config.batch}")
        for slot, name in enumerate(policies):
            if name not in table:
                raise KeyError(f"unknown policy {name!r}; have {sorted(table)}")
            # Not hash(name): Python randomises string hashing per process, which would make the
            # run irreproducible across invocations while looking deterministic.
            rng = np.random.default_rng(seed * 31 + 101 * slot)
            chosen = first.copy()
            common = {"screen_id": str(corpus.screen_id[target]),
                      "pmid": str(corpus.pmid[target]),
                      "phenotype": str(corpus.phenotype[target]),
                      "seed": seed, "policy": name, "pool": config.pool,
                      "block_cell_line": config.block_cell_line,
                      "n_genes": n, "n_hits": int(total), "n_donors": len(donors)}

            def _row(acquired, cycle: int, _common=common) -> dict:
                return {**_common, "cycle": cycle, "acquired": len(acquired),
                        "recall": float(labels[acquired].sum()) / total,
                        "chance": len(acquired) / n}

            if config.record_cycles:
                rows.append(_row(chosen, 1))
            for cycle in range(2, config.cycles + 1):
                pool = np.setdiff1d(np.arange(n), chosen)
                mean = spread = None
                if name != "random":
                    alpha = config.model.tune(X[chosen], labels[chosen], seed)
                    mean, spread = config.model.fit_predict(X, labels, chosen, alpha, seed)
                picked = table[name](rng, pool, config.batch, X=X, chosen=chosen, mean=mean,
                                     spread=spread, donor_hits=H, donor_tested=T,
                                     observed=labels)
                chosen = np.concatenate([chosen, picked])
                if config.record_cycles:
                    rows.append(_row(chosen, cycle))
            if len(set(chosen.tolist())) != len(chosen):
                raise RuntimeError(f"{name} acquired a gene twice on screen "
                                   f"{corpus.screen_id[target]}; the budget would be overstated")
            if not config.record_cycles:
                final = _row(chosen, config.cycles)
                del final["cycle"]
                rows.append(final)
            if config.record_picks:
                # Pool indices are meaningless outside this call, so the trace is written in the
                # corpus's own gene indexing. Acquisition ORDER is preserved: a trace sorted for
                # tidiness would lose which cycle each gene arrived in.
                traces.append({**common, "picks": idx[chosen].astype(np.int32)})
    return (rows, traces) if config.record_picks else rows
