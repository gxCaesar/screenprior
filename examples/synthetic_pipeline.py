"""Run the installed CLI on generated inputs, without downloading research data."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

POLICIES = (
    "random", "greedy", "topuncertain", "softuncertain", "coreset", "badge",
    "task-thompson", "task-thompson/permuted", "task-thompson/uniform",
)


def make_inputs(out: Path) -> tuple[Path, Path, Path]:
    rng = np.random.default_rng(0)
    orcs = out / "orcs"
    orcs.mkdir()
    genes = [f"G{i:04d}" for i in range(400)]
    programme = rng.integers(0, 3, size=len(genes))
    columns = ["#SCREEN_ID", "SOURCE_ID", "SCORES_SIZE", "NUMBER_OF_HITS", "CELL_LINE",
               "PHENOTYPE", "ORGANISM_OFFICIAL"]
    index_rows = ["\t".join(columns)]
    for screen in range(6):
        hits = np.where(programme == screen % 3, rng.random(400) < 0.30,
                        rng.random(400) < 0.03)
        lines = ["#SCREEN_ID\tIDENTIFIER_ID\tOFFICIAL_SYMBOL\tSCORE.1\tHIT"]
        lines.extend(f"{200 + screen}\tX\t{gene}\t0.0\t{'YES' if hit else 'NO'}"
                     for gene, hit in zip(genes, hits))
        (orcs / f"BIOGRID-ORCS-SCREEN_{200 + screen}-2.0.18.screen.tab.txt").write_text(
            "\n".join(lines) + "\n", encoding="utf-8")
        publication = "synthetic1" if screen < 2 else f"synthetic{screen}"
        index_rows.append("\t".join([str(200 + screen), publication, "400", str(int(hits.sum())),
                                     "synthetic-cell", "synthetic-phenotype", "H. sapiens"]))
    index = out / "SCREEN_INDEX.index.tab.txt"
    index.write_text("\n".join(index_rows) + "\n", encoding="utf-8")
    dep_rng = np.random.default_rng(1)
    measured = [f"G{i:04d}" for i in range(350)][::-1] + [f"X{i:03d}" for i in range(10)]
    values = dep_rng.normal(0.0, 0.3, size=(80, len(measured)))
    values[3, 7] = np.nan
    frame = pd.DataFrame(values, columns=[f"{g} ({1000 + i})" for i, g in enumerate(measured)],
                         index=[f"ACH-{i:06d}" for i in range(80)])
    frame.index.name = "DepMap_ID"
    depmap = out / "CRISPR_gene_effect.csv"
    frame.to_csv(depmap)
    return orcs, index, depmap


def cli(*args: str) -> str:
    result = subprocess.run([sys.executable, "-m", "screenprior.cli", *args],
                            check=True, text=True, capture_output=True)
    print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, file=sys.stderr, end="")
    return result.stdout


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="New directory for generated files")
    args = parser.parse_args()
    if args.out.exists():
        parser.error("--out must name a new directory")
    args.out.mkdir(parents=True)
    out = args.out.resolve()
    orcs, index, depmap = make_inputs(out)
    corpus, aligned = out / "corpus.npz", out / "corpus.depmap.npz"
    features, effects = out / "features.npy", out / "gene-effect.npy"
    results = out / "results.csv"
    cli("build-corpus", "--orcs", str(orcs), "--index", str(index), "--out", str(corpus),
        "--min-tested", "100", "--min-hits", "5")
    cli("build-static", "--depmap", str(depmap), "--corpus", str(corpus), "--out-corpus",
        str(aligned), "--features", str(features), "--gene-effect", str(effects))
    policy_args = [arg for policy in POLICIES for arg in ("--policy", policy)]
    cli("run", "--corpus", str(aligned), "--features", str(features), "--gene-effect",
        str(effects), "--out", str(results), *policy_args, "--batch", "8", "--cycles", "3",
        "--seeds", "2", "--pool", "full", "--min-pool", "100", "--min-hits", "5")
    report = cli("report", "--results", str(results), "--against", "badge", "--n-candidates", "1")
    (out / "report.txt").write_text(report, encoding="utf-8")
    rows = pd.read_csv(results)
    summary = {"synthetic": True, "targets": int(rows.screen_id.nunique()),
               "represented_genes": len(np.load(effects)), "result_rows": len(rows),
               "acquisitions_per_row": sorted(rows.acquired.unique().tolist()),
               "policies": list(POLICIES), "seeds": sorted(rows.seed.unique().tolist())}
    if (summary["targets"], summary["represented_genes"], summary["result_rows"],
            summary["acquisitions_per_row"]) != (5, 350, 90, [24]):
        raise RuntimeError(f"unexpected synthetic output shape: {summary}")
    if not np.isfinite(rows.recall).all():
        raise RuntimeError("the example produced a non-finite recall")
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
