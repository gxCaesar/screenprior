"""Saved-result checks use reported rows and never fit or infer acquisition models."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from analysis_functions import signed_rank_p
from report_assayformer import DEFAULT_DATA, main, report_saved_rows
from scipy import stats
from study_metadata import validate_metadata


@pytest.fixture
def stored():
    """Read the complete released result tables with publication strings preserved."""
    return (
        pd.read_csv(DEFAULT_DATA / "screenprior_rows.csv", dtype={"screen": str, "pmid": str}),
        pd.read_csv(DEFAULT_DATA / "assayformer_matched_return_rows.csv", dtype={"screen": str}),
        pd.read_csv(DEFAULT_DATA / "screens.csv", dtype={"screen": str, "publication_group": str}),
    )


def test_saved_rows_reproduce_reported_comparison(stored):
    """The full saved grid must recover the original estimand and both uncertainty intervals."""
    result = report_saved_rows(*stored)
    assert result["validated_cells"] == {"task_thompson": 200, "assayformer": 400}
    assert result["outcome"] == "M3"
    assert result["D"] == pytest.approx(0.4188133815109931, abs=1e-12)
    assert result["ci95_screens"] == pytest.approx([-0.29366646881532504, 1.1829430147254447], abs=1e-12)
    assert result["ci95_publications"] == pytest.approx([-0.3271181401024106, 1.3281908603230743], abs=1e-12)
    assert result["paired_ef"] == {"won": 11, "p": 0.47490501403808594}
    assert result["paired_frac_hits"]["p"] == pytest.approx(0.4523754119873047, abs=1e-12)
    assert result["never_read"]["D"] == pytest.approx(0.012064129227818259, abs=1e-12)
    nested = result["uncertainty"]["nested_never_read"]
    assert nested["screens"] == 11 and nested["publication_groups"] == 7
    assert nested["ci95_publications"] == pytest.approx([-0.700528524843567, 0.8369136937525777], abs=1e-12)
    assert "figure-derived" in nested["role"]
    assert result["model_inference_performed"] is False


def test_missing_seed_refuses_incomplete_assayformer(stored):
    """A nine-seed phase cannot be reported as a complete ten-seed comparison."""
    route, af, meta = stored
    with pytest.raises(ValueError, match="expected 400 unique cells"):
        report_saved_rows(route, af.loc[af.seed != 9], meta)


def test_wrong_matched_pool_refuses_before_inference(stored):
    """Consistent seed rows from another candidate pool must fail the paired pool audit."""
    route, af, meta = stored
    af = af.copy()
    af.loc[(af.screen == meta.screen.iloc[0]) & af.pool.eq("matched"), "n_genes"] += 1
    with pytest.raises(ValueError, match="matched n_genes differs"):
        report_saved_rows(route, af, meta)
    joint_route = route.copy()
    joint_route.loc[joint_route.screen == meta.screen.iloc[0], "n_genes"] += 1
    with pytest.raises(ValueError, match="differs from screen metadata"):
        report_saved_rows(joint_route, af, meta)


def test_duplicate_and_missing_cell_are_not_hidden_by_grouping(stored):
    """Replacing one phase cell by a duplicate keeps 400 rows but corrupts pairing."""
    route, af, meta = stored
    defective = pd.concat([af.iloc[:-1], af.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate_rows"):
        report_saved_rows(route, defective, meta)


def test_wrong_ef_denominator_refuses_native_counts(stored):
    """Native EF uses in-library acquisitions, not all nominated genes."""
    route, af, meta = stored
    af = af.copy()
    native = af.pool.eq("native")
    af.loc[native, "ef"] = af.loc[native, "found"] / (
        1000 * af.loc[native, "total_hits"] / af.loc[native, "library_size"])
    with pytest.raises(ValueError, match="EF disagrees"):
        report_saved_rows(route, af, meta)


def test_extraneous_route_row_is_not_silently_filtered(stored):
    """A complete target table plus an unrelated arm is not exactly 200 supplied cells."""
    route, af, meta = stored
    with pytest.raises(ValueError, match="exactly 200"):
        report_saved_rows(pd.concat([route, route.iloc[[0]]], ignore_index=True), af, meta)


def test_metadata_order_and_nested_membership_are_fixed(stored):
    """Changing screen order changes finite bootstrap draws and must be explicit failure."""
    meta = stored[2]
    with pytest.raises(ValueError, match="original order"):
        validate_metadata(meta.iloc[::-1])
    changed = meta.copy()
    changed.loc[0, ["development_read", "never_read"]] = [False, True]
    with pytest.raises(ValueError, match="membership changed"):
        validate_metadata(changed)


def test_integer_counts_and_nonfinite_values_are_refused(stored):
    """Fractions in counts and missing EF cannot produce plausible aggregate statistics."""
    route, af, meta = stored
    changed = af.copy()
    changed["n_in_library"] = changed["n_in_library"].astype(float)
    changed.loc[0, "n_in_library"] += 0.5
    with pytest.raises(ValueError, match="must be integers"):
        report_saved_rows(route, changed, meta)
    changed = af.copy()
    changed.loc[0, "ef"] = np.nan
    with pytest.raises(ValueError, match="missing or nonfinite"):
        report_saved_rows(route, changed, meta)


def test_signed_rank_null_and_tie_rule():
    """The null outcome is defined, and zero/tied contrasts retain the reported normal rule."""
    assert signed_rank_p(np.zeros(20)) == 1.0
    differences = np.tile([0.0, 1.0, 1.0, -2.0], 10)
    expected = stats.wilcoxon(differences, zero_method="wilcox", method="approx").pvalue
    assert signed_rank_p(differences) == expected


def test_cli_writes_scientific_json_without_private_inputs(tmp_path):
    """The runnable adapter emits results without source paths or execution receipts."""
    out = tmp_path / "comparison.json"
    assert main(["--out", str(out)]) == 0
    text = out.read_text()
    result = json.loads(text)
    assert result["outcome"] == "M3"
    for private in (str(Path.home()), "experiments/", "run_receipt", "sha256", "approval"):
        assert private not in text
    with pytest.raises(SystemExit) as error:
        main(["--out", str(out)])
    assert error.value.code == 1
