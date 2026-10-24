"""One external target at a time, with publication-blocked donors and label-free controls.

Construction never appends another external target to the donor corpus. The unchanged harness
performs its own publication blocking before the donor embedder sees any matrix. Only the
continuous-stream random control is substituted, with a fresh closure for each seed. The other
eight registry entries, their nine-arm slots, and the shared opening remain unchanged.

N is reported as n_genes, H as n_hits, and every acquired-hit count is recovered from the retained
gene-order trace. No function writes a result artifact or launches a process.
"""

# Preserve the established ValueError API and dictionary construction.
# ruff: noqa: C408, TRY004

from __future__ import annotations

import re

import numpy as np

from .corpus import GENEDISCO_SIBLING_PMIDS, Corpus
from .features import corpus_embedder
from .harness import RunConfig, run_screen
from .policies import POLICIES, donor_agreement
from .random_control import corrected_random_policy

CORE_POLICIES = (
    "random", "greedy", "topuncertain", "softuncertain", "coreset", "badge",
    "task-thompson", "task-thompson/permuted", "task-thompson/uniform",
)
SEEDS = tuple(range(10))
BATCH, CYCLES = 64, 10


def _identity(value):
    value = str(value).strip()
    for prefix in ("pmid_", "doi_", "https://doi.org/", "http://dx.doi.org/"):
        if value.startswith(prefix):
            value = value[len(prefix):]
            break
    preprint = re.search(r"10\.1101/\d{4}\.\d{2}\.\d{2}\.\d+", value)
    return (preprint.group() if preprint else value).casefold()


def _binary_vector(values, n_genes, name):
    values = np.asarray(values)
    if values.shape != (n_genes,) or not np.isin(values, (0, 1)).all():
        raise ValueError(f"{name} must be a binary vector in the corpus gene order")
    return values.astype(bool, copy=True)


def external_corpus(base: Corpus, tested, hit, pub_id: str, screen_id: str, metadata,
                    blocked_donor_publications) -> Corpus:
    """Retained frozen ORCS donors plus exactly one external target, at the last column.

    `blocked_donor_publications` is explicit, including an explicit empty set for the author-block
    sensitivity. Fixed GeneDisco exclusions are retained. Target metadata may supply actual PMID,
    DOI, cell_line and phenotype; pub_id is the stable assembly identifier. Existing target
    publication/screen identities and unknown requested donor blocks are refused.
    """
    if getattr(base, "_a1_external_target", None) is not None:
        raise ValueError("an external target is already in the supplied base corpus")
    if not str(pub_id).strip() or not str(screen_id).strip():
        raise ValueError("external target needs a publication and screen identifier")
    tested = _binary_vector(tested, len(base.genes), "target tested")
    hit = _binary_vector(hit, len(base.genes), "target hit")
    if (hit & ~tested).any():
        raise ValueError("external target has a hit on an untested gene")
    if blocked_donor_publications is None:
        raise ValueError("supply an explicit blocked-donor publication set")
    if isinstance(blocked_donor_publications, str):
        raise ValueError("blocked-donor publications must be a set, not one string")
    base_ids = {_identity(p) for p in base.pmid}
    target_ids = {_identity(pub_id)} | {
        _identity(metadata[k]) for k in ("pmid", "doi") if metadata.get(k)
    }
    if base_ids & target_ids:
        raise ValueError("external target publication is already in the base corpus")
    if str(screen_id) in set(map(str, base.screen_id)):
        raise ValueError("external target screen is already in the base corpus")
    blocked = {_identity(p) for p in blocked_donor_publications}
    if blocked - base_ids:
        raise ValueError(f"unknown blocked donor publications: {sorted(blocked - base_ids)}")
    removed = blocked | {_identity(p) for p in GENEDISCO_SIBLING_PMIDS}
    keep = np.array([_identity(p) not in removed for p in base.pmid], dtype=bool)
    if not keep.any():
        raise ValueError("no donor screens remain after the publication exclusions")
    target_pmid = str(metadata.get("pmid") or pub_id)
    result = Corpus(
        genes=base.genes.copy(),
        screen_id=np.concatenate([base.screen_id.astype(str)[keep], [str(screen_id)]]),
        pmid=np.concatenate([base.pmid.astype(str)[keep], [target_pmid]]),
        cell_line=np.concatenate([base.cell_line.astype(str)[keep],
                                  [str(metadata.get("cell_line", ""))]]),
        phenotype=np.concatenate([base.phenotype.astype(str)[keep],
                                  [str(metadata.get("phenotype", ""))]]),
        hit=np.column_stack([base.hit[:, keep], hit]),
        tested=np.column_stack([base.tested[:, keep], tested]),
    )
    result._a1_external_target = (str(pub_id), str(screen_id))
    return result


def _context(corpus, target, gene_effect, pool):
    if target != len(corpus.screen_id) - 1:
        raise ValueError("the single external target must be the final corpus column")
    effect = np.asarray(gene_effect)
    if effect.shape != (len(corpus.genes),) or not np.isfinite(effect).all():
        raise ValueError("gene effect must be finite and aligned to the corpus genes")
    if pool not in ("full", "non-essential"):
        raise ValueError("pool must be 'full' or 'non-essential'")
    keep = corpus.tested[:, target].copy()
    if pool == "non-essential":
        keep &= effect >= -0.5
    indices = np.flatnonzero(keep)
    if not len(indices) or not corpus.hit[indices, target].any():
        raise ValueError("external target pool is empty or has no hits")
    return indices, corpus.donors_for(target), effect


def _common(corpus, target, indices, donors, pub_id, pool):
    return dict(pub_id=str(pub_id), screen_id=str(corpus.screen_id[target]),
                pmid=str(corpus.pmid[target]), phenotype=str(corpus.phenotype[target]),
                pool=pool, block_cell_line=False, n_genes=len(indices),
                n_hits=int(corpus.hit[indices, target].sum()), n_donors=len(donors))


def _outcome(corpus, target, picks, common):
    acquired_hits = int(corpus.hit[picks, target].sum())
    return dict(**common, acquired=len(picks), acquired_hits=acquired_hits,
                recall=acquired_hits / common["n_hits"], chance=len(picks) / common["n_genes"])


def after_cycle3_diagnostics(corpus, target, traces, *, pub_id):
    """Post hoc beta=1 donor concentration after 192 ScreenPrior acquisitions, without RNG.

    This output describes the posterior *after* acquiring cycle 3, rather than the posterior
    used to choose that cycle.
    Weights are over donor screens, as in task_thompson, and remain traceable to their identifiers.
    """
    donors = corpus.donors_for(target)
    diagnostics = []
    for trace in traces:
        if trace["policy"] != "task-thompson":
            continue
        if (str(trace["screen_id"]) != str(corpus.screen_id[target]) or
                str(trace["pmid"]) != str(corpus.pmid[target])):
            raise ValueError("cycle-3 trace belongs to another external target")
        picks = np.asarray(trace["picks"])
        if len(picks) < BATCH * 3:
            raise ValueError("cycle-3 diagnostic needs at least 192 ordered acquisitions")
        chosen = picks[:BATCH * 3]
        if (len(np.unique(chosen)) != BATCH * 3 or chosen.min() < 0 or
                chosen.max() >= len(corpus.genes) or not corpus.tested[chosen, target].all()):
            raise ValueError("cycle-3 acquisitions must be distinct tested target genes")
        z = donor_agreement(corpus.hit[np.ix_(chosen, donors)],
                            corpus.tested[np.ix_(chosen, donors)],
                            corpus.hit[chosen, target].astype(np.float64))
        weights = np.exp(z - z.max())
        weights /= weights.sum()
        diagnostics.append(dict(
            pub_id=str(pub_id), screen_id=str(corpus.screen_id[target]),
            pmid=str(corpus.pmid[target]), pool=trace["pool"], seed=trace["seed"],
            policy="task-thompson", diagnostic_time="after_cycle3", acquired=BATCH * 3,
            beta=1.0, n_donors=len(donors),
            effective_donor_count=float(1.0 / np.sum(weights * weights)),
            picks=chosen.astype(np.int32), donor_indices=donors.astype(np.int32),
            donor_screen_ids=corpus.screen_id[donors].astype(str),
            donor_publication_ids=corpus.pmid[donors].astype(str),
            donor_agreement_z=z, donor_weights=weights,
        ))
    return diagnostics


def run_one_target(corpus, target, features, gene_effect, *, pub_id, pool="full",
                   embed_donors=None, model=None):
    """Run the fixed nine arms, 64x10, seeds 0..9, on an `external_corpus` result.

    The default is the frozen rank-64 corpus embedding. A supplied callable and model are useful
    for synthetic interface tests; production passes the frozen static block/model. Embeddings
    are cached only inside this target call, after the unchanged harness supplies its blocked
    donor matrix. Registry mutation is local and replaces only `random`.
    """
    indices, donors, _ = _context(corpus, target, gene_effect, pool)
    static = np.asarray(features)
    if (static.ndim != 2 or static.shape[0] != len(corpus.genes) or
            not np.isfinite(static).all()):
        raise ValueError("static features must be finite and aligned to the corpus genes")
    embed = corpus_embedder(rank=64, seed=0) if embed_donors is None else embed_donors
    if not callable(embed):
        raise ValueError("donor embedding must be a callable applied after blocking")
    cached_input = cached_embedding = None

    def cached_embedder(donor_hits):
        nonlocal cached_input, cached_embedding
        if cached_input is None:
            cached_input = donor_hits.copy()
            cached_embedding = embed(donor_hits)
        elif not np.array_equal(cached_input, donor_hits):
            raise ValueError("donor matrix changed inside one external-target run")
        return cached_embedding

    common = _common(corpus, target, indices, donors, pub_id, pool)
    all_rows, all_traces = [], []
    for seed in SEEDS:
        config_args = dict(batch=BATCH, cycles=CYCLES, seeds=(seed,), pool=pool,
                           record_cycles=True, record_picks=True)
        if model is not None:
            config_args["model"] = model
        config = RunConfig(**config_args)
        if len(indices) < config.batch:
            raise ValueError("external target pool is smaller than the shared opening batch")
        registry = dict(POLICIES)
        registry["random"] = corrected_random_policy(seed, len(indices), BATCH)
        rows, traces = run_screen(
            corpus, target, static, gene_effect, policies=CORE_POLICIES, config=config,
            embed_donors=cached_embedder, registry=registry)
        if not rows:
            return dict(rows=[], traces=[], diagnostics=[], status="not_eligible", **common)
        by_policy = {t["policy"]: t for t in traces}
        for trace in traces:
            trace.update(pub_id=str(pub_id), acquisition_design="sequential-core")
            trace["acquired_labels"] = corpus.hit[trace["picks"], target].astype(np.uint8)
        for row in rows:
            picks = by_policy[row["policy"]]["picks"][:row["acquired"]]
            expected = _outcome(corpus, target, picks, common)
            if any(row[k] != expected[k] for k in ("n_genes", "n_hits", "acquired",
                                                  "n_donors", "recall", "chance")):
                raise ValueError("core row disagrees with its exact acquired-gene trace")
            all_rows.append({**row, "pub_id": str(pub_id),
                             "acquired_hits": expected["acquired_hits"],
                             "acquisition_design": "sequential-core"})
        all_traces.extend(traces)
    diagnostics = after_cycle3_diagnostics(corpus, target, all_traces, pub_id=pub_id)
    return dict(rows=all_rows, traces=all_traces, diagnostics=diagnostics, status="complete")


def label_free_baselines(corpus, target, gene_effect, *, pub_id, pool="full", budget=640,
                         batch=64, seeds=SEEDS):
    """Two donor-only/static scores, each pure-ranked once and after the shared opening.

    Pure controls have seed=-1 and cycle=0, one row per publication/pool. Shared-opening controls
    acquire the same initial 64 as the core, then the highest-ranked unacquired 576. Both score
    definitions are fixed: sum donor hits / max(sum donor tested, 1), and negative DepMap gene
    effect. Ties follow the unchanged corpus gene order. No surrogate or target label is read to
    choose a gene. Small budget/batch arguments support hand-computed synthetic checks.
    """
    indices, donors, effect = _context(corpus, target, gene_effect, pool)
    if (budget < 1 or batch < 1 or budget % batch or budget < batch or
            int(budget) != budget or int(batch) != batch or len(indices) < budget):
        raise ValueError("need an integral batch/budget and enough distinct candidate genes")
    seeds = tuple(seeds)
    if len(set(seeds)) != len(seeds) or any(s < 0 or int(s) != s for s in seeds):
        raise ValueError("shared-opening seeds must be distinct nonnegative integers")
    scores = {
        "corpus-frequency": (corpus.hit[:, donors].sum(axis=1) /
                             np.maximum(corpus.tested[:, donors].sum(axis=1), 1)),
        "depmap-essentiality": -effect,
    }
    common = _common(corpus, target, indices, donors, pub_id, pool)
    rows, traces = [], []
    for name, score in scores.items():
        ranking = indices[np.argsort(-score[indices], kind="stable")]
        choices = [(f"{name}/pure", -1, 0, "pure-ranking", ranking[:budget])]
        for seed in seeds:
            first = indices[np.random.default_rng(seed).choice(len(indices), batch, replace=False)]
            remainder = ranking[~np.isin(ranking, first)][:budget - batch]
            picks = np.concatenate([first, remainder])
            choices.append((f"{name}/shared-opening", seed, budget // batch,
                            "shared-opening-ranking", picks))
        for policy, seed, cycle, design, picks in choices:
            result = _outcome(corpus, target, picks, common)
            result.update(policy=policy, seed=seed, cycle=cycle, acquisition_design=design)
            rows.append(result)
            traces.append(dict(**common, policy=policy, seed=seed, acquisition_design=design,
                               picks=picks.astype(np.int32),
                               acquired_labels=corpus.hit[picks, target].astype(np.uint8)))
    return dict(rows=rows, traces=traces, diagnostics=[], status="complete")
