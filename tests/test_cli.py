"""build-corpus -> run -> report, end to end on files written by the test.

This is the "someone else can install it and it works" evidence. It runs the three commands in
order through the real entry point, on a corpus small enough to finish in seconds, and checks the
two places the CLI is supposed to refuse rather than default.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from screenprior.cli import app
from screenprior.corpus import Corpus

runner = CliRunner()

INDEX_COLUMNS = ["#SCREEN_ID", "SOURCE_ID", "SCORES_SIZE", "NUMBER_OF_HITS", "CELL_LINE",
                 "PHENOTYPE", "ORGANISM_OFFICIAL"]
SCREEN_HEADER = "#SCREEN_ID\tIDENTIFIER_ID\tOFFICIAL_SYMBOL\tSCORE.1\tHIT"


def build_inputs(tmp_path, n_genes=400, n_screens=6, seed=0, organism="H. sapiens"):
    """Six screens over four hundred genes, two of them sharing a publication."""
    rng = np.random.default_rng(seed)
    orcs = tmp_path / "orcs"
    orcs.mkdir()
    genes = [f"G{i:04d}" for i in range(n_genes)]
    programme = rng.integers(0, 3, size=n_genes)
    index_rows = []
    for s in range(n_screens):
        member = programme == (s % 3)
        hit = np.where(member, rng.random(n_genes) < 0.30, rng.random(n_genes) < 0.03)
        lines = [SCREEN_HEADER]
        for gene, call in zip(genes, hit):
            lines.append(f"{200 + s}\tX\t{gene}\t{rng.normal():.4f}\t{'YES' if call else 'NO'}")
        (orcs / f"BIOGRID-ORCS-SCREEN_{200 + s}-2.0.18.screen.tab.txt").write_text(
            "\n".join(lines) + "\n")
        # Screens 0 and 1 deliberately share a publication, so blocking has something to remove.
        index_rows.append([str(200 + s), "pmid1" if s < 2 else f"pmid{s}", str(n_genes),
                           str(int(hit.sum())), "K562", "proliferation", organism])
    index = tmp_path / "index.tab.txt"
    index.write_text("\n".join(["\t".join(INDEX_COLUMNS)]
                               + ["\t".join(r) for r in index_rows]) + "\n")
    np.save(tmp_path / "features.npy",
            np.column_stack([rng.normal(size=n_genes), programme.astype(float)]))
    np.save(tmp_path / "gene_effect.npy", rng.normal(size=n_genes))
    return orcs, index


def test_build_run_report_end_to_end(tmp_path):
    orcs, index = build_inputs(tmp_path)

    built = runner.invoke(app, ["build-corpus", "--orcs", str(orcs), "--index", str(index),
                                "--out", str(tmp_path / "corpus.npz"),
                                "--min-tested", "100", "--min-hits", "5"])
    assert built.exit_code == 0, built.output
    assert "publications" in built.output
    assert (tmp_path / "corpus.npz").exists()

    ran = runner.invoke(app, ["run", "--corpus", str(tmp_path / "corpus.npz"),
                              "--features", str(tmp_path / "features.npy"),
                              "--gene-effect", str(tmp_path / "gene_effect.npy"),
                              "--out", str(tmp_path / "results.csv"),
                              "--policy", "random", "--policy", "task-thompson",
                              "--batch", "8", "--cycles", "3", "--seeds", "2", "--pool", "full",
                              "--min-pool", "100", "--min-hits", "5"])
    assert ran.exit_code == 0, ran.output
    # Six screens, but two share a publication, so five targets.
    assert "5 target screens" in ran.output

    reported = runner.invoke(app, ["report", "--results", str(tmp_path / "results.csv"),
                                   "--against", "random", "--n-candidates", "7"])
    assert reported.exit_code == 0, reported.output
    payload = json.loads(reported.output[:reported.output.index("\n\n")])
    assert payload["against"] == "random"
    assert payload["arm"] == "task-thompson"
    assert 0.0 <= payload["wins"] <= 1.0
    assert payload["inflation_bar"] > 0


def write_depmap(tmp_path, n_genes=400, n_lines=80, seed=1):
    """A 21Q4-layout gene-effect file covering the first 350 corpus genes and ten it lacks, with
    the corpus genes in reverse order and every fifteenth one shifted below the -0.5 cut."""
    rng = np.random.default_rng(seed)
    measured = [f"G{i:04d}" for i in range(350)][::-1] + [f"X{i:03d}" for i in range(10)]
    values = rng.normal(0.0, 0.3, size=(n_lines, len(measured)))
    for column, gene in enumerate(measured):
        if gene.startswith("G") and int(gene[1:]) % 15 == 0:
            values[:, column] -= 1.0
    values[3, 7] = np.nan
    frame = pd.DataFrame(values, columns=[f"{g} ({1000 + k})" for k, g in enumerate(measured)],
                         index=[f"ACH-{i:06d}" for i in range(n_lines)])
    frame.index.name = "DepMap_ID"
    path = tmp_path / "CRISPR_gene_effect.csv"
    frame.to_csv(path)
    return path


def test_build_static_writes_what_run_takes(tmp_path):
    """build-corpus -> build-static -> run, with the arrays checked against the file itself.

    The mean gene effect is recomputed here from the CSV by column name, an independent route to
    the same number, and the pool `run` scored is checked against the cut applied to it.
    """
    orcs, index = build_inputs(tmp_path)
    depmap = write_depmap(tmp_path)
    assert runner.invoke(app, ["build-corpus", "--orcs", str(orcs), "--index", str(index),
                               "--out", str(tmp_path / "corpus.npz"), "--min-tested", "100",
                               "--min-hits", "5"]).exit_code == 0
    built = runner.invoke(app, ["build-static", "--depmap", str(depmap),
                                "--corpus", str(tmp_path / "corpus.npz"),
                                "--out-corpus", str(tmp_path / "corpus.depmap.npz"),
                                "--features", str(tmp_path / "static.npy"),
                                "--gene-effect", str(tmp_path / "ge.npy")])
    assert built.exit_code == 0, built.output
    assert "350 of 400 corpus genes" in built.output

    restricted = Corpus.load(tmp_path / "corpus.depmap.npz")
    whole = Corpus.load(tmp_path / "corpus.npz")
    block, effect = np.load(tmp_path / "static.npy"), np.load(tmp_path / "ge.npy")
    assert restricted.genes.tolist() == [f"G{i:04d}" for i in range(350)]
    assert block.shape == (350, 64) and block.dtype == np.float32
    assert effect.shape == (350,) and effect.dtype == np.float32
    np.testing.assert_array_equal(restricted.hit, whole.hit[:350])
    np.testing.assert_array_equal(restricted.tested, whole.tested[:350])
    assert restricted.screen_id.tolist() == whole.screen_id.tolist()

    frame = pd.read_csv(depmap, index_col=0)
    by_name = {c.split(" (")[0]: frame[c].fillna(0.0).mean() for c in frame.columns}
    np.testing.assert_allclose(effect, [by_name[g] for g in restricted.genes], atol=1e-6)

    ran = runner.invoke(app, ["run", "--corpus", str(tmp_path / "corpus.depmap.npz"),
                              "--features", str(tmp_path / "static.npy"),
                              "--gene-effect", str(tmp_path / "ge.npy"),
                              "--out", str(tmp_path / "results.csv"),
                              "--policy", "random", "--policy", "task-thompson",
                              "--batch", "8", "--cycles", "3", "--seeds", "2",
                              "--min-pool", "100", "--min-hits", "5"])
    assert ran.exit_code == 0, ran.output
    assert "5 target screens" in ran.output
    results = pd.read_csv(tmp_path / "results.csv")
    # Every synthetic screen tests every gene, so the non-essential pool is the 350 measured genes
    # less the 24 shifted below the cut.
    assert set(results.n_genes) == {sum(by_name[g] >= -0.5 for g in restricted.genes)} == {326}

    # The corpus build-static read is not aligned to these arrays, and run says so.
    wrong = runner.invoke(app, ["run", "--corpus", str(tmp_path / "corpus.npz"),
                                "--features", str(tmp_path / "static.npy"),
                                "--gene-effect", str(tmp_path / "ge.npy"),
                                "--out", str(tmp_path / "wrong.csv")])
    assert wrong.exit_code != 0
    assert "relabel every result" in " ".join(wrong.output.replace("│", " ").split())


def test_build_static_drops_genedisco_siblings_still_in_its_input(tmp_path):
    """The archived corpus is written before the sibling publications are removed, so the step
    that writes the corpus run reads has to remove them too or the run gains their screens."""
    rng = np.random.default_rng(2)
    hit = rng.random((400, 3)) < 0.1
    Corpus(np.array([f"G{i:04d}" for i in range(400)]), np.array(["1", "2", "2421"]),
           np.array(["pmidA", "pmidB", "35113687"]), np.array(["K562"] * 3),
           np.array(["proliferation"] * 3), hit, np.ones_like(hit)).save(tmp_path / "c.npz")
    built = runner.invoke(app, ["build-static", "--depmap", str(write_depmap(tmp_path)),
                                "--corpus", str(tmp_path / "c.npz"),
                                "--out-corpus", str(tmp_path / "r.npz"),
                                "--features", str(tmp_path / "f.npy"),
                                "--gene-effect", str(tmp_path / "g.npy")])
    assert built.exit_code == 0, built.output
    assert Corpus.load(tmp_path / "r.npz").screen_id.tolist() == ["1", "2"]


def test_build_static_refuses_a_renamed_depmap_file_and_writes_nothing(tmp_path):
    """The orientation of the matrix is read from DepMap's filename; under another name the file
    could be either, so nothing is guessed and nothing is written."""
    orcs, index = build_inputs(tmp_path)
    runner.invoke(app, ["build-corpus", "--orcs", str(orcs), "--index", str(index),
                        "--out", str(tmp_path / "corpus.npz"), "--min-tested", "100",
                        "--min-hits", "5"])
    renamed = tmp_path / "depmap_21q4.csv"
    write_depmap(tmp_path).rename(renamed)
    result = runner.invoke(app, ["build-static", "--depmap", str(renamed),
                                 "--corpus", str(tmp_path / "corpus.npz"),
                                 "--out-corpus", str(tmp_path / "r.npz"),
                                 "--features", str(tmp_path / "f.npy"),
                                 "--gene-effect", str(tmp_path / "g.npy")])
    assert result.exit_code != 0
    # rich draws the error in a box whose edges land inside wrapped phrases.
    assert "decided by the filename" in " ".join(result.output.replace("│", " ").split())
    assert not any((tmp_path / name).exists() for name in ("r.npz", "f.npy", "g.npy"))


def test_run_appends_the_per_target_corpus_block_by_default(tmp_path, monkeypatch):
    """The command once called run_screen without embed_donors, so every CLI run used the static
    block alone and was not the policy the paper reports."""
    from screenprior import cli

    orcs, index = build_inputs(tmp_path)
    runner.invoke(app, ["build-corpus", "--orcs", str(orcs), "--index", str(index),
                        "--out", str(tmp_path / "corpus.npz"), "--min-tested", "100",
                        "--min-hits", "5"])
    seen = []
    real = cli.run_screen

    def spy(*args, embed_donors=None, **kwargs):
        seen.append(embed_donors)
        return real(*args, embed_donors=embed_donors, **kwargs)

    monkeypatch.setattr(cli, "run_screen", spy)
    common = ["run", "--corpus", str(tmp_path / "corpus.npz"),
              "--features", str(tmp_path / "features.npy"),
              "--gene-effect", str(tmp_path / "gene_effect.npy"),
              "--out", str(tmp_path / "results.csv"), "--policy", "task-thompson",
              "--batch", "8", "--cycles", "2", "--seeds", "1", "--pool", "full",
              "--min-pool", "100", "--min-hits", "5"]
    assert runner.invoke(app, common).exit_code == 0
    assert seen and all(callable(e) for e in seen)
    seen.clear()
    assert runner.invoke(app, common + ["--corpus-rank", "0"]).exit_code == 0
    assert seen and all(e is None for e in seen)


def test_build_corpus_refuses_an_index_of_the_wrong_organism(tmp_path):
    """The mouse archive's index carries the same filename as the human one, and the copy named
    for the release on the analysis host was the mouse index."""
    orcs, index = build_inputs(tmp_path, organism="M. musculus")
    built = runner.invoke(app, ["build-corpus", "--orcs", str(orcs), "--index", str(index),
                                "--out", str(tmp_path / "corpus.npz"),
                                "--min-tested", "100", "--min-hits", "5"])
    assert built.exit_code != 0
    assert "M. musculus" in built.output
    assert not (tmp_path / "corpus.npz").exists()


def test_report_refuses_without_the_number_of_candidates_searched(tmp_path):
    """A default of one candidate set the most lenient selection-inflation bar, 0.0057 where seven
    candidates give 0.0093 on the published table, so the count has to be stated."""
    orcs, index = build_inputs(tmp_path)
    runner.invoke(app, ["build-corpus", "--orcs", str(orcs), "--index", str(index),
                        "--out", str(tmp_path / "corpus.npz"), "--min-tested", "100",
                        "--min-hits", "5"])
    runner.invoke(app, ["run", "--corpus", str(tmp_path / "corpus.npz"),
                        "--features", str(tmp_path / "features.npy"),
                        "--gene-effect", str(tmp_path / "gene_effect.npy"),
                        "--out", str(tmp_path / "results.csv"), "--policy", "random",
                        "--policy", "task-thompson", "--batch", "8", "--cycles", "2",
                        "--seeds", "1", "--pool", "full", "--min-pool", "100", "--min-hits", "5"])
    result = runner.invoke(app, ["report", "--results", str(tmp_path / "results.csv"),
                                 "--against", "random"])
    assert result.exit_code != 0
    assert "n-candidates" in result.output


def test_report_refuses_without_a_named_comparator(tmp_path):
    """A margin with no stated comparator is not a result, so `--against` has no default."""
    result = runner.invoke(app, ["report", "--results", str(tmp_path / "nothing.csv")])
    assert result.exit_code != 0
    assert "against" in result.output.lower()


def test_run_refuses_misaligned_features_rather_than_scoring_them(tmp_path):
    """Silently misaligning features to genes would relabel every result."""
    orcs, index = build_inputs(tmp_path)
    runner.invoke(app, ["build-corpus", "--orcs", str(orcs), "--index", str(index),
                        "--out", str(tmp_path / "corpus.npz"),
                        "--min-tested", "100", "--min-hits", "5"])
    np.save(tmp_path / "short.npy", np.zeros((10, 2)))
    result = runner.invoke(app, ["run", "--corpus", str(tmp_path / "corpus.npz"),
                                 "--features", str(tmp_path / "short.npy"),
                                 "--gene-effect", str(tmp_path / "gene_effect.npy"),
                                 "--out", str(tmp_path / "r.csv")])
    assert result.exit_code != 0
    # rich wraps the panel and its box characters break phrases across lines, so match a short
    # fragment that survives wrapping rather than the whole sentence.
    assert "relabel every result" in " ".join(result.output.split())


def test_build_corpus_refuses_a_screen_the_index_promises_but_does_not_ship(tmp_path):
    orcs, index = build_inputs(tmp_path)
    next(orcs.glob("*.screen.tab.txt")).unlink()
    result = runner.invoke(app, ["build-corpus", "--orcs", str(orcs), "--index", str(index),
                                 "--out", str(tmp_path / "c.npz"),
                                 "--min-tested", "100", "--min-hits", "5"])
    assert result.exit_code != 0
    assert "is absent" in result.output


def test_run_refuses_to_write_an_empty_results_file(tmp_path):
    """Every screen below the minimums must stop the run, not produce a 0-row csv and exit 0."""
    orcs, index = build_inputs(tmp_path)
    runner.invoke(app, ["build-corpus", "--orcs", str(orcs), "--index", str(index),
                        "--out", str(tmp_path / "corpus.npz"),
                        "--min-tested", "100", "--min-hits", "5"])
    result = runner.invoke(app, ["run", "--corpus", str(tmp_path / "corpus.npz"),
                                 "--features", str(tmp_path / "features.npy"),
                                 "--gene-effect", str(tmp_path / "gene_effect.npy"),
                                 "--out", str(tmp_path / "empty.csv"),
                                 "--policy", "random", "--batch", "8", "--cycles", "3",
                                 "--pool", "full"])          # min_pool stays at its 5,000 default
    assert result.exit_code != 0
    assert "fell below the minimums" in " ".join(result.output.split())
    assert not (tmp_path / "empty.csv").exists()


@pytest.mark.parametrize("command", ["build-corpus", "build-static", "run", "report"])
def test_every_command_documents_itself(command):
    result = runner.invoke(app, [command, "--help"])
    assert result.exit_code == 0
    assert len(result.output) > 200
