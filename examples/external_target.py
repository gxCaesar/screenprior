"""Run one generated external target against a fixed synthetic donor corpus."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from screenprior.corpus import Corpus
from screenprior.external import (
    CORE_POLICIES,
    external_corpus,
    label_free_baselines,
    run_one_target,
)
from screenprior.features import corpus_embedder
from screenprior.random_control import continuous_random_indices


def make_inputs():
    """Keep untested genes distinct from tested non-hits in donors and target."""
    n_genes = 5200
    rng = np.random.default_rng(11)
    tested = rng.random((n_genes, 4)) >= 0.10
    hit = (rng.random((n_genes, 4)) < 0.08) & tested
    donors = Corpus(
        genes=np.array([f"G{i:05d}" for i in range(n_genes)]),
        screen_id=np.array(["donor-a1", "donor-a2", "donor-b", "donor-c"]),
        pmid=np.array(["DONOR-A", "DONOR-A", "DONOR-B", "DONOR-C"]),
        cell_line=np.array(["CELL-A", "CELL-A", "CELL-B", "CELL-C"]),
        phenotype=np.array(["synthetic response"] * 4), hit=hit, tested=tested)
    target_tested = np.arange(n_genes) % 101 != 0
    target_hit = (np.arange(n_genes) % 17 == 0) & target_tested
    features = rng.normal(size=(n_genes, 3))
    gene_effect = rng.normal(0, 0.3, size=n_genes)
    return donors, target_tested, target_hit, features, gene_effect


def save_traces(path: Path, traces: list[dict]) -> None:
    np.savez_compressed(
        path, screen_id=np.array([t["screen_id"] for t in traces]),
        seed=np.array([t["seed"] for t in traces]), policy=np.array([t["policy"] for t in traces]),
        picks=np.stack([t["picks"] for t in traces]),
        acquired_labels=np.stack([t["acquired_labels"] for t in traces]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="New directory for generated files")
    args = parser.parse_args()
    if args.out.exists():
        parser.error("--out must name a new directory")
    args.out.mkdir(parents=True)
    out = args.out.resolve()
    donors, target_tested, target_hit, features, gene_effect = make_inputs()
    blocked_publications = {"DONOR-C"}
    corpus = external_corpus(
        donors, target_tested, target_hit, pub_id="EXTERNAL", screen_id="external-target",
        metadata={"cell_line": "CELL-EXTERNAL", "phenotype": "synthetic response"},
        blocked_donor_publications=blocked_publications)
    target = len(corpus.screen_id) - 1
    if corpus.screen_id.tolist() != ["donor-a1", "donor-a2", "donor-b", "external-target"]:
        raise RuntimeError("unexpected donor blocking or external-target construction")
    donors.save(out / "donors.npz")
    np.savez_compressed(out / "target_and_features.npz", genes=donors.genes,
                        tested=target_tested, hit=target_hit, features=features,
                        gene_effect=gene_effect)

    # The callable is applied only to retained donors. Its rank is small for
    # this generated corpus; the adapter's default rank for real inputs is 64.
    sequential = run_one_target(
        corpus, target, features, gene_effect, pub_id="EXTERNAL", pool="full",
        embed_donors=corpus_embedder(rank=2, seed=0))
    if sequential["status"] != "complete":
        raise RuntimeError(f"synthetic target is not eligible: {sequential['status']}")
    controls = label_free_baselines(corpus, target, gene_effect, pub_id="EXTERNAL", pool="full")
    rows = pd.DataFrame(sequential["rows"])
    control_rows = pd.DataFrame(controls["rows"])
    diagnostic_rows = pd.DataFrame([
        {key: value for key, value in item.items() if not isinstance(value, np.ndarray)}
        for item in sequential["diagnostics"]])
    rows.to_csv(out / "sequential.csv", index=False)
    control_rows.to_csv(out / "controls.csv", index=False)
    diagnostic_rows.to_csv(out / "diagnostics.csv", index=False)
    save_traces(out / "sequential_traces.npz", sequential["traces"])
    save_traces(out / "control_traces.npz", controls["traces"])
    diagnostic = sequential["diagnostics"]
    np.savez_compressed(
        out / "donor_weights.npz", seed=np.array([d["seed"] for d in diagnostic]),
        donor_screen_ids=diagnostic[0]["donor_screen_ids"],
        donor_publication_ids=diagnostic[0]["donor_publication_ids"],
        donor_weights=np.stack([d["donor_weights"] for d in diagnostic]),
        donor_agreement_z=np.stack([d["donor_agreement_z"] for d in diagnostic]))

    indices = np.flatnonzero(target_tested)
    for trace in sequential["traces"]:
        if len(np.unique(trace["picks"])) != 640 or not target_tested[trace["picks"]].all():
            raise RuntimeError("synthetic trace contains duplicate or untested acquisitions")
        if trace["policy"] == "random":
            expected = indices[continuous_random_indices(len(indices), trace["seed"])]
            np.testing.assert_array_equal(trace["picks"], expected)
    final = rows.loc[rows.cycle.eq(10)]
    if (len(rows), len(final), len(control_rows), len(diagnostic_rows)) != (900, 90, 22, 10):
        raise RuntimeError("unexpected synthetic output dimensions")
    if not np.isfinite(rows[["recall", "chance"]].to_numpy()).all():
        raise RuntimeError("the example produced non-finite metrics")
    summary = {
        "synthetic": True, "external_targets": 1, "represented_genes": len(corpus.genes),
        "retained_donor_screens": len(corpus.donors_for(target)),
        "retained_donor_publications": len(set(corpus.pmid[:-1])),
        "blocked_donor_publications": sorted(blocked_publications),
        "target_tested": int(target_tested.sum()), "target_hits": int(target_hit.sum()),
        "donor_untested_entries": int((~donors.tested).sum()),
        "target_untested_genes": int((~target_tested).sum()),
        "sequential_rows": len(rows), "final_rows": len(final),
        "sequential_traces": len(sequential["traces"]), "control_rows": len(control_rows),
        "diagnostic_rows": len(diagnostic_rows), "final_acquisitions_per_trace": 640,
        "seeds": sorted(rows.seed.unique().tolist()), "policies": list(CORE_POLICIES),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
