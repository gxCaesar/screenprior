"""Interface checks with synthetic outcomes; these tests do not fit acquisition models."""

import copy
from itertools import product

import numpy as np
import pandas as pd
import pytest
from adjudicate_a1 import CORE_POLICIES, adjudicate_primary
from report_secondary import read_diagnostics
from run_study import execute
from study_data import AUTHOR_BLOCKS, load_author_blocks, report_metadata

from screenprior.corpus import Corpus
from screenprior.external import external_corpus


def mock_payload():
    """One synthetic publication is enough to exercise all three condition paths."""
    n = 6000
    corpus = Corpus(np.array([f"g{i:05d}" for i in range(n)]), np.array(["d1", "d2"]),
                    np.array(["donor1", "donor2"]), np.array(["", ""]), np.array(["", ""]),
                    np.zeros((n, 2), dtype=bool), np.ones((n, 2), dtype=bool))
    hit = np.zeros((n, 1), dtype=bool)
    hit[:60] = True
    return {"corpus": corpus, "features": np.zeros((n, 64), dtype=np.float32),
            "gene_effect": np.zeros(n), "metadata": [{"pub_id": "pmid_test",
            "pmid": "test", "screen_id": "A1-pmid_test", "window": "window_wide",
            "grade": "A"}], "tested": np.ones_like(hit), "hit": hit,
            "blocked": {"pmid_test": {"donor1"}}}


def mock_outcome(corpus, target, *, pub_id, pool, free=False):
    """Manufacture consistent interface rows and diagnostic weights, not policy predictions."""
    candidate = np.flatnonzero(corpus.tested[:, target])
    donors = corpus.donors_for(target)
    n, h = len(candidate), int(corpus.hit[candidate, target].sum())
    common = {"pub_id": pub_id, "screen_id": str(corpus.screen_id[target]),
              "pmid": str(corpus.pmid[target]), "pool": pool, "block_cell_line": False,
              "n_genes": n, "n_hits": h, "n_donors": len(donors)}
    if free:
        grid = [(f"{name}/{kind}", seed, 0 if kind == "pure" else 10)
                for name in ("corpus-frequency", "depmap-essentiality")
                for kind in ("pure", "shared-opening")
                for seed in ((-1,) if kind == "pure" else range(10))]
    else:
        grid = list(product(CORE_POLICIES, range(10), range(1, 11)))
    rows = []
    for policy, seed, cycle in grid:
        acquired = 640 if free else 64 * cycle
        found = 30 if free else 3 * cycle
        rows.append({**common, "policy": policy, "seed": seed, "cycle": cycle,
                     "acquired": acquired, "acquired_hits": found,
                     "recall": found / h, "chance": acquired / n})
    trace_grid = [(arm, seed) for arm, seed, _ in grid]
    traces = [{**common, "policy": arm, "seed": seed,
               "mock_wiring_only": True, "picks": np.arange(640, dtype=np.int32)}
              for arm, seed in dict.fromkeys(trace_grid)]
    weights = np.ones(len(donors)) / len(donors)
    diagnostics = [] if free else [
        {**common, "policy": "task-thompson", "seed": seed, "acquired": 192,
         "diagnostic_time": "after_cycle3", "beta": 1.0,
         "donor_agreement_z": np.zeros(len(donors)), "donor_weights": weights,
         "effective_donor_count": float(len(donors)), "picks": np.arange(192)}
        for seed in range(10)]
    return {"status": "complete", "rows": rows, "traces": traces, "diagnostics": diagnostics}


@pytest.fixture
def mocked_api(monkeypatch):
    """Replace only the two functions that would fit/score policies."""
    calls = []

    def core(corpus, target, features, effect, **kwargs):
        calls.append(("core", kwargs["pool"], len(corpus.donors_for(target))))
        return mock_outcome(corpus, target, **kwargs)

    def free(corpus, target, effect, **kwargs):
        calls.append(("free", kwargs["pool"], len(corpus.donors_for(target))))
        return mock_outcome(corpus, target, free=True, **kwargs)

    monkeypatch.setattr("run_study.external.run_one_target", core)
    monkeypatch.setattr("run_study.external.label_free_baselines", free)
    return calls


def test_serial_adapter_wires_three_conditions_and_reporter(tmp_path, mocked_api):
    """Block lists, condition tags, integer counts and saved diagnostics reach the reporters."""
    out = tmp_path / "fresh"
    result = execute(mock_payload(), out)
    assert result["core_rows"] == 2700 and result["free_rows"] == 66
    assert mocked_api == [(kind, pool, donors) for pool, donors in (
        ("full", 1), ("non-essential", 1), ("full", 2)) for kind in ("core", "free")]
    primary = pd.read_csv(out / "primary_rows.csv", dtype={"pub_id": str, "pmid": str})
    assert adjudicate_primary(primary, ["pmid_test"])["n_rows_validated"] == 900
    diagnostics, paths = read_diagnostics(out / "targets")
    assert len(diagnostics) == 10 and len(paths) == 1
    status = pd.read_csv(out / "condition_status.csv")
    assert status.status.eq("completed").all()
    with pytest.raises(FileExistsError):
        execute(mock_payload(), out)


def test_ineligible_nonessential_does_not_dispatch_a_model(tmp_path, mocked_api):
    """A pool below the unchanged hit floor remains explicitly omitted from that condition."""
    payload = mock_payload()
    payload["gene_effect"][:60] = -1
    result = execute(payload, tmp_path / "fresh")
    assert result["ineligible_conditions"] == 1
    assert len(mocked_api) == 4
    status = pd.read_csv(tmp_path / "fresh/condition_status.csv")
    omitted = status.loc[status.condition.eq("non-essential")].iloc[0]
    assert omitted.status == "not_eligible" and omitted.n_hits == 0


def test_missing_seed_is_refused():
    """A plausible nine-seed table cannot be called the ten-seed primary analysis."""
    payload = mock_payload()
    corpus = external_corpus(payload["corpus"], payload["tested"][:, 0], payload["hit"][:, 0],
                             "pmid_test", "A1-pmid_test", payload["metadata"][0], set())
    rows = mock_outcome(corpus, 2, pub_id="pmid_test", pool="full")["rows"]
    for row in rows:
        row.update(author_blocking=True)
    with pytest.raises(ValueError, match="coverage"):
        adjudicate_primary([row for row in rows if row["seed"] != 9], ["pmid_test"])


def test_source_metadata_and_blocks_have_exact_40_coverage():
    """The public metadata retains selection order and covers all selected donor exclusions."""
    selected, metadata = report_metadata()
    blocks = load_author_blocks(AUTHOR_BLOCKS, selected.pub_id.tolist())
    assert len(selected) == len(metadata) == len(blocks) == 40
    assert sum(map(len, blocks.values())) == 33
    assert "library_type" not in selected and metadata.phenotype.eq("").all()


def test_metadata_order_and_incomplete_blocks_are_refused(tmp_path):
    """An otherwise valid metadata table or exclusion list cannot silently attach to wrong targets."""
    selected, metadata = report_metadata()
    metadata.iloc[::-1].to_csv(tmp_path / "reversed.csv", index=False)
    with pytest.raises(ValueError, match="slot order"):
        report_metadata(metadata_path=tmp_path / "reversed.csv")
    blocks = pd.read_csv(AUTHOR_BLOCKS, dtype=str)
    blocks.iloc[:-1].to_csv(tmp_path / "incomplete.csv", index=False)
    with pytest.raises(ValueError, match="exact selected target order"):
        load_author_blocks(tmp_path / "incomplete.csv", selected.pub_id.tolist())


def test_wrong_model_denominator_stops_and_leaves_partial_status(tmp_path, mocked_api, monkeypatch):
    """An API defect cannot silently change the target universe or produce a complete primary."""
    from run_study import external

    original = external.run_one_target

    def wrong(*args, **kwargs):
        result = copy.deepcopy(original(*args, **kwargs))
        result["rows"][0]["n_genes"] += 1
        return result

    monkeypatch.setattr(external, "run_one_target", wrong)
    out = tmp_path / "failed"
    with pytest.raises(RuntimeError, match="failed condition"):
        execute(mock_payload(), out)
    statuses = pd.read_csv(out / "condition_status.csv")
    assert statuses.status.tolist() == ["failed", "not_run", "not_run"]
    assert not (out / "primary_rows.csv").exists()
