"""Command line for the four things that need to happen in order.

    screenprior build-corpus --orcs <dir> --index <file> --out corpus.npz
    screenprior build-static --depmap CRISPR_gene_effect.csv --corpus corpus.npz
                             --out-corpus corpus.depmap.npz --features features.npy
                             --gene-effect ge.npy
    screenprior run          --corpus corpus.depmap.npz --features features.npy
                             --gene-effect ge.npy
    screenprior report       --results results.csv --against badge

The CLI refuses rather than defaults in the two places where a default would be a scientific
choice: it will not run without publication blocking, and it will not report a margin without
naming what the margin is against.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import typer

from screenprior.corpus import GENEDISCO_SIBLING_PMIDS, Corpus, read_screen
from screenprior.features import corpus_embedder, depmap_matrix, static_block
from screenprior.harness import RunConfig, run_screen
from screenprior.metrics import signed_rank_p

app = typer.Typer(add_completion=False, help=__doc__)


@app.command("build-corpus")
def build_corpus(
    orcs: Path = typer.Option(..., help="Directory of BIOGRID-ORCS-SCREEN_*.screen.tab.txt files"),
    index: Path = typer.Option(..., help="The release's SCREEN_INDEX .index.tab.txt"),
    out: Path = typer.Option(..., help="Where to write the .npz"),
    min_tested: int = typer.Option(12_000, help="Genome-scale cut on the tested universe"),
    min_hits: int = typer.Option(30, help="Screens with fewer hits cannot estimate anything"),
    max_hit_rate: float = typer.Option(0.20, help="A screen calling more than this is not measuring the same kind of thing"),
    drop_genedisco_siblings: bool = typer.Option(True, help="Remove the three publications whose screens are GeneDisco assays"),
    strict: bool = typer.Option(True, help="Refuse unparseable rows instead of relabelling them"),
    organism: str = typer.Option("H. sapiens", help="Keep screens whose ORGANISM_OFFICIAL is this"),
) -> None:
    """Parse the screens into a gene x screen hit matrix with its tested mask."""
    meta = pd.read_csv(index, sep="\t", dtype=str, na_filter=False)
    meta.columns = [c.lstrip("#") for c in meta.columns]
    # Human and mouse archives can share the SCREEN_INDEX filename. The mouse index has
    # 205 screens, all M. musculus, in release 2.0.18. Check the recorded organism rather than
    # assuming it from the filename.
    if "ORGANISM_OFFICIAL" not in meta:
        raise typer.BadParameter(f"{index} has no ORGANISM_OFFICIAL column, so the organism of "
                                 "its screens cannot be checked")
    present = sorted(set(meta.ORGANISM_OFFICIAL))
    meta = meta[meta.ORGANISM_OFFICIAL == organism]
    if meta.empty:
        raise typer.BadParameter(f"{index} lists no {organism} screens; it holds {present}. The "
                                 "human and mouse archives' indices share a filename")
    numeric = meta.assign(
        tested=pd.to_numeric(meta.SCORES_SIZE.str.replace(",", ""), errors="coerce"),
        hits=pd.to_numeric(meta.NUMBER_OF_HITS.str.replace(",", ""), errors="coerce"))
    keep = numeric[(numeric.tested >= min_tested) & (numeric.hits >= min_hits)
                   & (numeric.hits / numeric.tested <= max_hit_rate)]
    typer.echo(f"{len(keep):,} of {len(meta):,} screens pass the filters "
               f"({keep.SOURCE_ID.nunique():,} publications)")

    per_screen, kept = [], []
    for row in keep.itertuples():
        path = Path(orcs) / f"BIOGRID-ORCS-SCREEN_{row.SCREEN_ID}-2.0.18.screen.tab.txt"
        if not path.exists():
            raise typer.BadParameter(f"index lists screen {row.SCREEN_ID} but {path} is absent")
        symbols, hits, _ = read_screen(path, strict=strict)
        per_screen.append((symbols, hits))
        kept.append(row)

    genes = sorted({g for symbols, _ in per_screen for g in symbols})
    where = {g: i for i, g in enumerate(genes)}
    hit = np.zeros((len(genes), len(kept)), dtype=bool)
    tested = np.zeros_like(hit)
    for column, (symbols, calls) in enumerate(per_screen):
        rows = np.array([where[g] for g in symbols])
        tested[rows, column] = True
        hit[rows[calls], column] = True

    corpus = Corpus(np.array(genes), np.array([r.SCREEN_ID for r in kept]),
                    np.array([r.SOURCE_ID for r in kept]),
                    np.array([r.CELL_LINE for r in kept]),
                    np.array([r.PHENOTYPE for r in kept]), hit, tested)
    if drop_genedisco_siblings:
        before = corpus.hit.shape[1]
        corpus = corpus.drop_pmids(GENEDISCO_SIBLING_PMIDS)
        typer.echo(f"dropped {before - corpus.hit.shape[1]} screens sharing a GeneDisco PMID")
    corpus.save(out)
    typer.echo(f"wrote {out}: {corpus.hit.shape[0]:,} genes x {corpus.hit.shape[1]:,} screens")


@app.command("build-static")
def build_static(
    depmap: Path = typer.Option(..., help="The DepMap gene-effect CSV under the name DepMap gave it (CRISPR_gene_effect.csv for 21Q4); its layout is read from the filename"),
    corpus: Path = typer.Option(..., help="A .npz written by build-corpus"),
    out_corpus: Path = typer.Option(..., help="Where to write the corpus restricted to the genes DepMap measures. This, not the input, is the --corpus that run takes"),
    features: Path = typer.Option(..., help="Where to write the static block, (n_genes, 64) float32"),
    gene_effect: Path = typer.Option(..., help="Where to write the mean gene effect, (n_genes,) float32"),
    drop_genedisco_siblings: bool = typer.Option(True, help="Remove the three publications whose screens are GeneDisco assays, if the input corpus still holds them"),
) -> None:
    """Write the static block and the mean gene effect, one row per corpus gene DepMap measures.

    The three outputs share one gene order and are what run takes. A corpus gene DepMap does not
    measure is dropped from the corpus rather than given a row of zeros, which is what the reported
    runs did, so run must be given --out-corpus and not the corpus this command read.
    """
    data = Corpus.load(corpus)
    if drop_genedisco_siblings:
        before = data.hit.shape[1]
        data = data.drop_pmids(GENEDISCO_SIBLING_PMIDS)
        typer.echo(f"dropped {before - data.hit.shape[1]} screens sharing a GeneDisco PMID")
    try:
        values, symbols = depmap_matrix(depmap)
        rows, block, effect = static_block(data.genes, values, symbols)
    except (OSError, ValueError) as exc:
        # Nothing is written on a refusal, so a half-built set of inputs cannot be run by mistake.
        raise typer.BadParameter(str(exc)) from exc
    restricted = Corpus(data.genes[rows], data.screen_id, data.pmid, data.cell_line,
                        data.phenotype, data.hit[rows], data.tested[rows])
    restricted.save(out_corpus)
    np.save(features, block)
    np.save(gene_effect, effect)
    typer.echo(f"{len(rows):,} of {len(data.genes):,} corpus genes are in {Path(depmap).name} "
               f"({len(symbols):,} genes x {values.shape[1]:,} cell lines)")
    typer.echo(f"wrote {out_corpus}: {len(rows):,} genes x {restricted.hit.shape[1]:,} screens; "
               f"{features}: {block.shape}; {gene_effect}: {effect.shape}")


@app.command()
def run(
    corpus: Path = typer.Option(..., help="A .npz written by build-static (--out-corpus), or by build-corpus when the features come from elsewhere"),
    features: Path = typer.Option(..., help=".npy, (n_genes, d), aligned to the corpus gene order; build-static writes it"),
    gene_effect: Path = typer.Option(..., help=".npy, (n_genes,), DepMap mean gene effect -- used ONLY to define the pool, never as a label; build-static writes it"),
    out: Path = typer.Option(Path("results.csv")),
    policy: list[str] = typer.Option(["random", "badge", "task-thompson"],
                                     help="Repeatable"),
    batch: int = typer.Option(64),
    cycles: int = typer.Option(10),
    seeds: int = typer.Option(3, help="Seeds 0..n-1"),
    pool: str = typer.Option("non-essential", help="'non-essential' or 'full'"),
    min_pool: int = typer.Option(5_000, help="Screens with a smaller candidate pool are skipped"),
    min_hits: int = typer.Option(30, help="Screens with fewer hits cannot estimate anything"),
    block_cell_line: bool = typer.Option(False, help="Block donors sharing the target's cell line, on top of the mandatory publication blocking"),
    cell_lines: str = typer.Option("parent", help="'parent' merges sublines (HEK293T with HEK-293) before cell-line blocking; 'raw' compares the strings as recorded"),
    corpus_rank: int = typer.Option(64, help="Rank of the corpus embedding, rebuilt for each target from its blocked donors and appended to --features. 0 drops it, which is not the published policy"),
) -> None:
    """Run the policies on every target screen, one target per publication."""
    if cell_lines not in ("parent", "raw"):
        raise typer.BadParameter(f"--cell-lines is 'parent' or 'raw', not {cell_lines!r}")
    data = Corpus.load(corpus)
    if cell_lines == "parent":
        data = data.with_parent_cell_lines()
    X, effect = np.load(features), np.load(gene_effect)
    if len(X) != len(data.genes) or len(effect) != len(data.genes):
        raise typer.BadParameter(
            f"features {X.shape} and gene effect {effect.shape} must have one row per corpus gene "
            f"({len(data.genes)}); silently misaligning them would relabel every result")
    config = RunConfig(batch=batch, cycles=cycles, seeds=tuple(range(seeds)), pool=pool,
                       block_cell_line=block_cell_line, min_pool=min_pool, min_hits=min_hits)
    if corpus_rank < 0:
        raise typer.BadParameter(f"--corpus-rank must be 0 or positive, not {corpus_rank}")
    embedder = corpus_embedder(rank=corpus_rank) if corpus_rank else None
    targets = data.targets_one_per_publication()
    typer.echo(f"{len(targets)} target screens, one per publication")

    rows, skipped = [], 0
    with typer.progressbar(targets, label="screens") as progress:
        for target in progress:
            # The corpus block is a callable, not an array: built once over the whole corpus it
            # would contain the target's own hits. Omitting it, as this command once did, ran
            # every policy on the static block alone, which is not the method the paper reports.
            got = run_screen(data, int(target), X, effect, policies=tuple(policy), config=config,
                             embed_donors=embedder)
            rows.extend(got)
            skipped += not got
    if not rows:
        # Writing an empty results file and exiting 0 is the worst available behaviour: the next
        # command would fail on a parse error and the cause would be three steps upstream.
        raise typer.BadParameter(
            f"every one of the {len(targets)} target screens fell below the minimums "
            f"(min_pool={min_pool}, min_hits={min_hits}); nothing was written")
    pd.DataFrame(rows).to_csv(out, index=False)
    typer.echo(f"wrote {out}: {len(rows):,} rows, {skipped} screens below the minimums")


@app.command()
def report(
    results: Path = typer.Option(...),
    against: str = typer.Option(..., help="The comparator. Required: a margin with no named comparator is not a result"),
    arm: str = typer.Option("task-thompson"),
    n_candidates: int = typer.Option(..., min=1, help="How many candidates were searched, for the selection-inflation bar. Required: a default of one would set the most lenient bar"),
) -> None:
    """Paired margins with the selection-inflation bar computed on the PAIRED standard error."""
    frame = pd.read_csv(results)
    table = frame.pivot_table(index="screen_id", columns="policy", values="recall")
    for name in (arm, against):
        if name not in table:
            raise typer.BadParameter(f"{name!r} is not in {sorted(table.columns)}")
    d = (table[arm] - table[against]).dropna()
    se = float(d.std(ddof=1) / np.sqrt(len(d)))
    bar = float(np.sqrt(2 * np.log(max(n_candidates, 2))) * se)
    p = signed_rank_p(d)
    payload = {"arm": arm, "against": against, "screens": len(d),
               "margin": float(d.mean()), "paired_se": se, "wins": float((d > 0).mean()),
               "p": p, "inflation_bar": bar,
               "clears_bar": bool(d.mean() > bar), "n_candidates_searched": n_candidates}
    typer.echo(json.dumps(payload, indent=2))
    spread_se = float(table[arm].dropna().std(ddof=1) / np.sqrt(len(d)))
    typer.echo("\nsigma is the PAIRED standard error, not the spread of recall across screens --")
    typer.echo(f"the latter is about how much screens differ, and here its standard error is "
               f"{spread_se / se:.1f} times the paired one.")


if __name__ == "__main__":
    app()
