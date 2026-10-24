"""Report descriptive secondary endpoints from complete saved study rows.

Publication means precede every paired comparison, bootstrap and donor-concentration tercile.
Secondary contrasts are descriptive and have no success decision.
"""

# Keep the established validation API and dictionary construction.
# ruff: noqa: C408

from __future__ import annotations

import argparse
import json
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from adjudicate_a1 import SEEDS, signed_rank_p, strict_validate_primary
from scipy import stats
from study_data import public_report, report_metadata

CONDITIONS = {
    "primary": ("full", True),
    "non-essential": ("non-essential", True),
    "author-blocking-off": ("full", False),
}
FREE_POLICIES = (
    "corpus-frequency/pure", "depmap-essentiality/pure",
    "corpus-frequency/shared-opening", "depmap-essentiality/shared-opening",
)
COMPARATORS = ("badge", "task-thompson/uniform", "task-thompson/permuted", *FREE_POLICIES)
COMPOSITION = {
    "modality": ("modality", "library_type"),
    "cell_line": ("cell_line", "cell_line_or_type"),
    "library": ("library", "library_name"),
    "phenotype": ("phenotype",),
}
BOOTSTRAP = dict(draws=10000, seed=0, unit="publication", order="pub_id stable ascending",
                 interval="percentile 2.5 and 97.5")


def _boolean(value):
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, str) and value.strip().casefold() in {"true", "false"}:
        return value.strip().casefold() == "true"
    raise ValueError(f"explicit Boolean required, got {value!r}")


def _identities(frame, columns, name):
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(f"missing {name} columns: {sorted(missing)}")
    for column in columns:
        if frame[column].isna().any():
            raise ValueError(f"missing {name} identity: {column}")
        frame[column] = frame[column].map(str)
        if frame[column].map(lambda x: not x or x != x.strip()).any():
            raise ValueError(f"empty or untrimmed {name} identity: {column}")


def _counts(frame, columns, name):
    for column in columns:
        if column not in frame:
            raise ValueError(f"missing {name} count: {column}")
        try:
            values = pd.to_numeric(frame[column], errors="raise").to_numpy(dtype=float)
        except (ValueError, TypeError) as exc:
            raise ValueError(f"{name} {column} must be numeric") from exc
        if (not np.isfinite(values).all() or (values != np.floor(values)).any() or
                (np.abs(values) >= 2 ** 63).any()):
            raise ValueError(f"{name} {column} must contain finite integer counts")
        frame[column] = values.astype(np.int64)


def _known(value):
    return (value is not None and not pd.isna(value) and
            str(value).strip() not in {"", "[TBD]", "unknown", "UNKNOWN"})


def target_metadata(targets, manifest):
    """Join only final selections; preserve explicit composition values without inference."""
    selected, source = pd.DataFrame(targets).copy(), pd.DataFrame(manifest).copy()
    for name, frame in (("selection", selected), ("assembly manifest", source)):
        _identities(frame, ("pub_id", "grade", "window"), name)
        if frame.pub_id.duplicated().any():
            raise ValueError(f"duplicate {name} publication identifiers")
    if selected.empty or set(selected.pub_id) - set(source.pub_id):
        raise ValueError("selection must be nonempty and completely covered by the manifest")
    if "slot" in selected:
        _counts(selected, ("slot",), "selection")
        if selected.slot.tolist() != list(range(1, len(selected) + 1)):
            raise ValueError("selection must retain original slot order")
    source = source.set_index("pub_id")
    rows = []
    fields = ("mapped_scored", "mapped_hits", "represented_scored", "represented_hits")
    for choice in selected.to_dict("records"):
        pub = choice["pub_id"]
        row = source.loc[pub].to_dict()
        if any(row[k] != choice[k] for k in ("grade", "window")):
            raise ValueError(f"{pub}: selection grade/window disagrees with manifest")
        if row["grade"] not in {"A", "B"} or row["window"] not in {
                "window_wide", "window_strict"}:
            raise ValueError(f"{pub}: unresolved grade/window")
        numbers = pd.DataFrame([row])
        _counts(numbers, fields, "assembly")
        row.update({k: int(numbers.iloc[0][k]) for k in fields})
        mn, mh, rn, rh = (row[k] for k in fields)
        if (mn < 12000 or mh < 30 or mh / mn > .20 or rn < 5000 or rh < 30 or
                rn > mn or rh > mh or rh > rn):
            raise ValueError(f"{pub}: manifest violates mapped/represented eligibility")
        row.update(choice, pub_id=pub)
        row.update({k: int(numbers.iloc[0][k]) for k in fields})
        row["composition"] = {}
        for field, aliases in COMPOSITION.items():
            values = [(f"{origin}.{alias}", item[alias])
                      for origin, item in (("selection", choice), ("manifest", source.loc[pub]))
                      for alias in aliases if alias in item and _known(item[alias])]
            if len({str(value) for _, value in values}) > 1:
                raise ValueError(f"{pub}: conflicting explicit composition field {field}")
            provenance = []
            for origin, item in (("selection", choice), ("manifest", source.loc[pub])):
                for alias in aliases:
                    if alias in item:
                        provenance.append(dict(
                            field=f"{origin}.{alias}", source_file=(
                                str(item["composition_source_file"])
                                if _known(item.get("composition_source_file")) else None),
                            source_column=(str(item[alias + "_source_column"])
                                           if _known(item.get(alias + "_source_column")) else None),
                            metadata_status=(str(item[alias + "_metadata_status"])
                                             if _known(item.get(alias + "_metadata_status"))
                                             else None)))
            row["composition"][field] = dict(
                value=str(values[0][1]) if values else None,
                source_columns=[name for name, _ in values],
                status="known" if values else "unknown", provenance=provenance)
        rows.append(row)
    return rows


def _validate_statuses(rows, metadata):
    frame = pd.DataFrame(rows).copy()
    _identities(frame, ("pub_id", "condition", "pool", "status"), "condition status")
    keys = ["pub_id", "condition"]
    expected = set(product([r["pub_id"] for r in metadata], CONDITIONS))
    if (frame.duplicated(keys).any() or
            set(frame[keys].itertuples(index=False, name=None)) != expected):
        raise ValueError("condition status coverage must contain every publication x 3 conditions")
    _counts(frame, ("n_genes", "n_hits"), "condition status")
    if "author_blocking" not in frame:
        raise ValueError("condition status requires author_blocking")
    frame["author_blocking"] = frame.author_blocking.map(_boolean)
    allowed = {"completed", "not_eligible", "failed", "not_run"}
    meta = {r["pub_id"]: r for r in metadata}
    for row in frame.to_dict("records"):
        pub, condition = row["pub_id"], row["condition"]
        if (row["status"] not in allowed or
                (row["pool"], row["author_blocking"]) != CONDITIONS[condition]):
            raise ValueError(f"{pub}: condition status has invalid state/pool/blocking")
        n, h = row["n_genes"], row["n_hits"]
        if n < 0 or h < 0 or h > n:
            raise ValueError(f"{pub}: condition status N/H is invalid")
        if condition != "non-essential":
            if (n, h) != (meta[pub]["represented_scored"], meta[pub]["represented_hits"]):
                raise ValueError(f"{pub}: full-pool status N/H disagrees with manifest")
            if row["status"] == "not_eligible":
                raise ValueError("an eligible primary/full target cannot be marked not_eligible")
        elif n > meta[pub]["represented_scored"] or h > meta[pub]["represented_hits"]:
            raise ValueError(f"{pub}: non-essential N/H exceeds full-pool counts")
        eligible = n >= 5000 and h >= 30
        if ((row["status"] == "not_eligible" and eligible) or
                (row["status"] == "completed" and not eligible)):
            raise ValueError(f"{pub}: status does not match frozen N/H eligibility")
        if row["status"] != "not_run":
            donors = pd.DataFrame([row])
            _counts(donors, ("n_donors",), "condition status")
            if int(donors.n_donors.iloc[0]) < 1:
                raise ValueError("condition status requires at least one donor")
    if "n_donors" not in frame:
        frame["n_donors"] = np.nan
    return frame.sort_values(keys, kind="stable").reset_index(drop=True)


def _condition_rows(rows, statuses, metadata, *, free=False):
    name = "free controls" if free else "core"
    frame = pd.DataFrame(rows).copy()
    required = ("pub_id", "condition", "pool", "policy", "screen_id", "pmid")
    _identities(frame, required, name)
    for column in ("author_blocking", "block_cell_line"):
        if column not in frame:
            raise ValueError(f"{name} requires {column}")
        frame[column] = frame[column].map(_boolean)
    if frame.block_cell_line.any():
        raise ValueError(f"{name} must retain block_cell_line=False")
    completed = statuses.loc[statuses.status.eq("completed")]
    actual = set(frame[["pub_id", "condition"]].itertuples(index=False, name=None))
    expected = set(completed[["pub_id", "condition"]].itertuples(index=False, name=None))
    if actual != expected:
        raise ValueError(f"{name} condition coverage differs from completed status records")
    meta = {r["pub_id"]: r for r in metadata}
    for condition, (pool, blocking) in CONDITIONS.items():
        group = frame.loc[frame.condition.eq(condition)].copy()
        pubs = sorted(completed.loc[completed.condition.eq(condition), "pub_id"])
        if not pubs:
            continue
        if not group.pool.eq(pool).all() or not group.author_blocking.eq(blocking).all():
            raise ValueError(f"{name} {condition}: pool/author-blocking mismatch")
        if not free:
            normalized = group.assign(pool="full", author_blocking=True)
            checked = strict_validate_primary(normalized, pubs)
            for col in ("seed", "cycle", "n_genes", "n_hits", "n_donors", "acquired",
                        "acquired_hits", "recall", "chance"):
                # Retain the original condition flags after using the common core-grid validator.
                group.loc[:, col] = pd.to_numeric(group[col], errors="raise")
            if len(checked) != len(group):
                raise ValueError("core validator changed coverage")
        else:
            _validate_free_grid(group, pubs)
        status_index = completed.loc[completed.condition.eq(condition)].set_index("pub_id")
        for pub, part in group.groupby("pub_id", sort=False):
            for col in ("n_genes", "n_hits", "n_donors"):
                if (part[col].nunique() != 1 or
                        int(part[col].iloc[0]) != int(status_index.loc[pub, col])):
                    raise ValueError(f"{name} {pub}: {col} differs from condition status")
            for col in ("grade", "window"):
                if col in part and not part[col].eq(meta[pub][col]).all():
                    raise ValueError(f"{name} {pub}: {col} differs from final metadata")
    _counts(frame, ("seed", "cycle", "n_genes", "n_hits", "n_donors", "acquired",
                    "acquired_hits"), name)
    return frame.sort_values(["condition", "pub_id", "policy", "seed", "cycle"],
                             kind="stable").reset_index(drop=True)


def _validate_free_grid(frame, pubs):
    _counts(frame, ("seed", "cycle", "n_genes", "n_hits", "n_donors", "acquired",
                    "acquired_hits"), "free controls")
    keys = ["pub_id", "policy", "seed", "cycle"]
    expected = {(p, arm, seed, 0 if arm.endswith("/pure") else 10)
                for p in pubs for arm in FREE_POLICIES
                for seed in ((-1,) if arm.endswith("/pure") else SEEDS)}
    if (frame.duplicated(keys).any() or
            set(frame[keys].itertuples(index=False, name=None)) != expected):
        raise ValueError("free-control cell coverage must be 22 observations "
                         "per completed condition")
    n, h, found = frame.n_genes, frame.n_hits, frame.acquired_hits
    if (frame.acquired.ne(640).any() or (n < 5000).any() or (h < 30).any() or
            (h > n).any() or (frame.n_donors < 1).any() or (found < 0).any() or
            (found > np.minimum(640, h)).any()):
        raise ValueError("free-control acquired/N/H counts are invalid")
    for col, expected_value in (("recall", found / h), ("chance", 640 / n)):
        if col not in frame:
            raise ValueError(f"free controls require {col}")
        values = pd.to_numeric(frame[col], errors="raise").to_numpy(dtype=float)
        if not np.isfinite(values).all() or not np.allclose(values, expected_value,
                                                          rtol=0, atol=1e-12):
            raise ValueError(f"free-control {col} disagrees with count denominator")


def _validate_diagnostics(rows, core):
    frame = pd.DataFrame(rows).copy()
    _identities(frame, ("pub_id", "condition", "policy", "pool", "screen_id", "pmid",
                        "diagnostic_time"), "after-cycle3 diagnostics")
    _counts(frame, ("seed", "acquired", "n_donors"), "after-cycle3 diagnostics")
    primary = core.loc[core.condition.eq("primary")]
    pubs = sorted(primary.pub_id.unique())
    keys = ["pub_id", "seed"]
    if (frame.duplicated(keys).any() or
            set(frame[keys].itertuples(index=False, name=None)) != set(product(pubs, SEEDS))):
        raise ValueError("primary after-cycle3 diagnostic coverage must be ten seeds/publication")
    if (not frame.condition.eq("primary").all() or not frame.pool.eq("full").all() or
            not frame.policy.eq("task-thompson").all() or not frame.acquired.eq(192).all() or
            not frame.diagnostic_time.eq("after_cycle3").all() or
            "author_blocking" not in frame or not frame.author_blocking.map(_boolean).all()):
        raise ValueError("cycle-3 diagnostic must be author-blocked primary after 192 acquisitions")
    by_pub = primary.groupby("pub_id", sort=False).first()
    values = []
    for row in frame.to_dict("records"):
        pub = row["pub_id"]
        if (row["screen_id"] != by_pub.loc[pub, "screen_id"] or
                row["pmid"] != by_pub.loc[pub, "pmid"] or
                row["n_donors"] != int(by_pub.loc[pub, "n_donors"])):
            raise ValueError(f"{pub}: cycle-3 identity/donor count disagrees with primary")
        d = int(row["n_donors"])
        weights = np.asarray(row.get("donor_weights"), dtype=float)
        z = np.asarray(row.get("donor_agreement_z"), dtype=float)
        effective = float(row.get("effective_donor_count", np.nan))
        if (weights.shape != (d,) or z.shape != (d,) or not np.isfinite(weights).all() or
                not np.isfinite(z).all() or (weights < 0).any() or row.get("beta") != 1.0 or
                not np.isclose(weights.sum(), 1, rtol=0, atol=1e-12) or
                not np.isfinite(effective) or not 1 - 1e-9 <= effective <= d + 1e-9):
            raise ValueError(f"{pub}: invalid after-cycle3 posterior")
        expected_weights = np.exp(z - z.max())
        expected_weights /= expected_weights.sum()
        if (not np.allclose(weights, expected_weights, rtol=0, atol=1e-12) or
                not np.isclose(effective, 1 / np.square(weights).sum(), rtol=0, atol=1e-9)):
            raise ValueError(f"{pub}: effective count disagrees with beta=1 donor posterior")
        picks = np.asarray(row.get("picks"))
        if (picks.shape != (192,) or not np.issubdtype(picks.dtype, np.integer) or
                (picks < 0).any() or len(np.unique(picks)) != 192):
            raise ValueError(f"{pub}: diagnostic requires 192 unique acquired gene rows")
        values.append(dict(pub_id=pub, seed=row["seed"], effective_donor_count=effective))
    return pd.DataFrame(values, columns=["pub_id", "seed", "effective_donor_count"])


def validate_secondary(core_rows, free_rows, condition_status, targets, manifest, diagnostics):
    """Validate exact declared coverage; no row is silently dropped or repaired."""
    metadata = target_metadata(targets, manifest)
    statuses = _validate_statuses(condition_status, metadata)
    core = _condition_rows(core_rows, statuses, metadata)
    free = _condition_rows(free_rows, statuses, metadata, free=True)
    final = core.loc[core.cycle.eq(10)]
    for (pub, condition), group in free.groupby(["pub_id", "condition"], sort=False):
        common = final.loc[final.pub_id.eq(pub) & final.condition.eq(condition)].iloc[0]
        for column in ("screen_id", "pmid", "n_genes", "n_hits", "n_donors"):
            if not group[column].eq(common[column]).all():
                raise ValueError(f"{pub}: free/core identity or denominator differs: {column}")
    for pub, group in final.groupby("pub_id", sort=False):
        if group.screen_id.nunique() != 1 or group.pmid.nunique() != 1:
            raise ValueError(f"{pub}: condition screen/PMID identity differs")
    publication_pmids = final.groupby("pub_id", sort=False).pmid.first()
    if publication_pmids.duplicated().any():
        raise ValueError("different publications cannot count the same PMID")
    for pub, group in statuses.groupby("pub_id", sort=False):
        counts = group.set_index("condition")
        base = counts.loc["primary", "n_donors"]
        if pd.notna(base):
            ne, off = (counts.loc[c, "n_donors"] for c in (
                "non-essential", "author-blocking-off"))
            if ((pd.notna(ne) and int(ne) != int(base)) or
                    (pd.notna(off) and int(off) < int(base))):
                raise ValueError(f"{pub}: donor counts contradict blocking conditions")
    effective = _validate_diagnostics(diagnostics, core)
    return dict(metadata=metadata, statuses=statuses, core=core, free=free, effective=effective)


def _interval(values):
    lower, upper = np.percentile(values, [2.5, 97.5])
    return [float(lower), float(upper)]


def summarize_contrast(publications, comparator):
    """Paired publication bootstrap; deterministic controls supply one genuine observation."""
    ordered = sorted(publications, key=lambda p: p["pub_id"])
    if not ordered:
        return dict(status="not_estimable", n_publications=0, publications=[])
    ours = np.array([p["arms"]["task-thompson"]["mean_acquired_hits"] for p in ordered])
    other = np.array([p["arms"][comparator]["mean_acquired_hits"] for p in ordered])
    h = np.array([p["n_hits"] for p in ordered])
    difference = ours - other
    margin = difference / h
    rng = np.random.default_rng(0)
    sampled = rng.integers(len(ordered), size=(10000, len(ordered)))
    wins, losses = int((margin > 0).sum()), int((margin < 0).sum())
    sign_n = wins + losses
    return dict(
        status="descriptive", comparator=comparator, n_publications=len(ordered),
        mean_screenprior_recall=float(np.mean(ours / h)),
        screenprior_recall_ci95=_interval((ours / h)[sampled].mean(1)),
        mean_comparator_recall=float(np.mean(other / h)),
        comparator_recall_ci95=_interval((other / h)[sampled].mean(1)),
        mean_margin=float(margin.mean()), margin_ci95=_interval(margin[sampled].mean(1)),
        median_margin=float(np.median(margin)),
        median_margin_ci95=_interval(np.median(margin[sampled], axis=1)),
        mean_screenprior_hits_per640=float(ours.mean()),
        screenprior_hits_ci95=_interval(ours[sampled].mean(1)),
        mean_comparator_hits_per640=float(other.mean()),
        comparator_hits_ci95=_interval(other[sampled].mean(1)),
        mean_additional_hits=float(difference.mean()),
        additional_hits_ci95=_interval(difference[sampled].mean(1)),
        median_additional_hits=float(np.median(difference)),
        median_additional_hits_ci95=_interval(np.median(difference[sampled], axis=1)),
        win_fraction=wins / len(ordered),
        win_fraction_ci95=_interval((margin[sampled] > 0).mean(1)),
        n_wins=wins, n_losses=losses, n_ties=len(ordered) - sign_n,
        signed_rank_p=signed_rank_p(margin), sign_test_n=sign_n,
        sign_binomial_p=(float(stats.binomtest(wins, sign_n, .5).pvalue) if sign_n else 1.0),
        sign_test_tie_rule="exclude zero margins; win fraction includes all publications",
        publications=[dict(pub_id=p["pub_id"], n_genes=p["n_genes"], n_hits=p["n_hits"],
                           screenprior_recall=float(a / hit), comparator_recall=float(b / hit),
                           margin=float(delta / hit), additional_hits=float(delta))
                      for p, a, b, hit, delta in zip(ordered, ours, other, h, difference)],
    )


def _endpoint_publications(validated):
    final = validated["core"].loc[validated["core"].cycle.eq(10)]
    all_rows = pd.concat([final, validated["free"]], ignore_index=True)
    metadata = {r["pub_id"]: r for r in validated["metadata"]}
    publications = []
    for (condition, pub), group in all_rows.groupby(["condition", "pub_id"], sort=True):
        identity = group.iloc[0]
        arms = {}
        for policy, arm in group.groupby("policy", sort=True):
            count = 1 if policy.endswith("/pure") else 10
            arms[policy] = dict(observations=count, seeds=sorted(arm.seed.tolist()),
                                mean_acquired_hits=int(arm.acquired_hits.sum()) / count,
                                mean_recall=int(arm.acquired_hits.sum()) /
                                (count * int(identity.n_hits)))
        meta = metadata[pub]
        publications.append(dict(
            pub_id=pub, pmid=str(identity.pmid), screen_id=str(identity.screen_id),
            condition=condition, pool=identity.pool, author_blocking=bool(identity.author_blocking),
            n_genes=int(identity.n_genes), n_hits=int(identity.n_hits),
            n_donors=int(identity.n_donors), budget=640, chance=640 / int(identity.n_genes),
            grade=meta["grade"], window=meta["window"], arms=arms,
        ))
    return publications


def _analysis(publications, expected, omissions):
    included = sorted(p["pub_id"] for p in publications)
    return dict(status="descriptive" if included else "not_estimable",
                n_expected_publications=len(expected), n_publications=len(included),
                included_publications=included, omitted_publications=omissions,
                contrasts={c: summarize_contrast(publications, c) for c in COMPARATORS})


def effective_donor_terciles(effective):
    """Match development qcut on ten-seed publication means; never invent rank cutoffs."""
    frame = pd.DataFrame(effective)
    means = frame.groupby("pub_id", sort=True).effective_donor_count.mean()
    values = [dict(pub_id=str(pub), mean_effective_donor_count=float(value))
              for pub, value in means.items()]
    edges = np.quantile(means.to_numpy(), np.linspace(0, 1, 4)).tolist() if len(means) else []
    common = dict(diagnostic_time="after_cycle3", acquired=192, seeds_averaged=list(SEEDS),
                  estimator="mean effective donor count across ten acquisition seeds",
                  method="pandas.qcut(q=3), development convention", actual_edges=edges,
                  n_publications=len(means), publications=values)
    if len(means) < 3:
        return dict(**common, status="not_estimable", reason="fewer than three publications",
                    actual_bins=[])
    try:
        bins, edges = pd.qcut(means, 3, retbins=True)
    except ValueError as exc:
        return dict(**common, status="not_estimable", reason=str(exc), actual_bins=[])
    common["actual_edges"] = [float(x) for x in edges]
    labels = ("most concentrated", "middle", "most diffuse")
    actual_bins = []
    for code, interval in enumerate(bins.cat.categories):
        pubs = sorted(bins.index[bins.cat.codes.eq(code)].tolist())
        actual_bins.append(dict(label=labels[code], interval=str(interval),
                                n_publications=len(pubs), publications=pubs))
    for row in values:
        row["tercile"] = labels[int(bins.cat.codes.loc[row["pub_id"]])]
    return dict(**common, status="descriptive", actual_bins=actual_bins)


def report_secondary(core_rows, free_rows, condition_status, targets, manifest, diagnostics):
    """Return all registered descriptive contrasts, sensitivity sets and explicit coverage."""
    valid = validate_secondary(core_rows, free_rows, condition_status, targets, manifest,
                               diagnostics)
    metadata, statuses = valid["metadata"], valid["statuses"]
    expected = sorted(r["pub_id"] for r in metadata)
    publications = _endpoint_publications(valid)
    conditions = {}
    for condition in CONDITIONS:
        selected = [p for p in publications if p["condition"] == condition]
        records = statuses.loc[statuses.condition.eq(condition)].to_dict("records")
        omitted = [dict(
            pub_id=r["pub_id"], status=r["status"], n_genes=int(r["n_genes"]),
            n_hits=int(r["n_hits"]), reason=next(
                (str(r[k]) for k in ("reason", "error") if _known(r.get(k))),
                "condition did not complete")) for r in records if r["status"] != "completed"]
        conditions[condition] = _analysis(selected, expected, omitted)
        conditions[condition]["status_counts"] = {
            name: sum(r["status"] == name for r in records)
            for name in ("completed", "not_eligible", "failed", "not_run")}
    primary = [p for p in publications if p["condition"] == "primary"]
    sensitivities = {}
    signal = []
    for meta in sorted(metadata, key=lambda r: r["pub_id"]):
        n, h = meta["mapped_scored"], meta["mapped_hits"]
        p = float(stats.binomtest(h, n, p=.05, alternative="greater").pvalue)
        signal.append(dict(pub_id=meta["pub_id"], n_genes=n, n_hits=h, p_value=p, included=p < .05))
    sets = {
        "strict-window": {r["pub_id"] for r in metadata if r["window"] == "window_strict"},
        "grade-A": {r["pub_id"] for r in metadata if r["grade"] == "A"},
        "minimal-signal": {r["pub_id"] for r in signal if r["included"]},
    }
    for name, ids in sets.items():
        rows = [p for p in primary if p["pub_id"] in ids]
        omissions = [r for r in conditions["primary"]["omitted_publications"] if r["pub_id"] in ids]
        sensitivities[name] = _analysis(rows, sorted(ids), omissions)
        sensitivities[name]["outside_sensitivity_publications"] = sorted(set(expected) - ids)
    sensitivities["minimal-signal"]["definition"] = dict(
        denominator="assembly mapped_scored", numerator="assembly mapped_hits",
        population="deduplicated mapped source screen before representation intersection",
        null_hit_probability=.05, test="one-sided exact binomial, alternative=greater",
        retain_if="P < 0.05", changes_primary=False, publications=signal)
    sensitivities["author-blocking-off"] = dict(
        conditions["author-blocking-off"], source_condition="author-blocking-off")
    terciles = effective_donor_terciles(valid["effective"])
    for band in terciles["actual_bins"]:
        subset = [p for p in primary if p["pub_id"] in band["publications"]]
        band["contrasts"] = {c: summarize_contrast(subset, c) for c in COMPARATORS}
    composition = {}
    for field in COMPOSITION:
        values = [dict(pub_id=r["pub_id"], **r["composition"][field])
                  for r in sorted(metadata, key=lambda r: r["pub_id"])]
        known = [r for r in values if r["status"] == "known"]
        counts = pd.Series([r["value"] for r in known], dtype=object).value_counts().sort_index()
        composition[field] = dict(n_publications=len(values), known_count=len(known),
                                  unknown_count=len(values) - len(known),
                                  counts={str(k): int(v) for k, v in counts.items()},
                                  publications=values)
    finished = statuses.status.isin(["completed", "not_eligible"]).all()
    return dict(
        validation_status="passed", coverage_status="complete" if finished else "partial",
        reporting_scope="predeclared descriptive secondary endpoints; uncorrected for multiplicity",
        interpretation="descriptive estimates only; no success rule",
        independent_unit="publication",
        n_selected_publications=len(metadata), n_core_rows_validated=len(valid["core"]),
        n_free_rows_validated=len(valid["free"]), n_condition_records=len(statuses),
        n_primary_diagnostic_records=len(valid["effective"]), bootstrap=BOOTSTRAP,
        seed_aggregation="ten seeds averaged within publication; pure controls one seed=-1 row",
        conditions=conditions, sensitivities=sensitivities, cycle3_terciles=terciles,
        composition=composition,
        composition_policy="explicit source values; unknowns never inferred",
        publications=publications,
    )


def read_diagnostics(root):
    """Read primary records only from the runner's one-directory-per-target structure."""
    paths = sorted(Path(root).glob("*/primary/after_cycle3.jsonl"))
    rows = []
    for path in paths:
        with path.open(encoding="utf-8") as fh:
            rows.extend(json.loads(line) for line in fh if line.strip())
    return rows, paths


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("core", "free", "status", "diagnostics-root", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--targets", type=Path,
                        default=Path(__file__).resolve().parent.parent / "preparation/targets.csv")
    parser.add_argument("--metadata", type=Path,
                        default=Path(__file__).resolve().with_name("study-metadata.csv"))
    args = parser.parse_args(argv)
    if args.out.exists():
        raise FileExistsError(f"secondary report already exists: {args.out}")
    targets, manifest = report_metadata(args.targets, args.metadata)
    core, free = (pd.read_csv(path, dtype={"pub_id": str, "screen_id": str, "pmid": str})
                  for path in (args.core, args.free))
    status = pd.read_csv(args.status, dtype={"pub_id": str})
    diagnostics, _ = read_diagnostics(args.diagnostics_root)
    report = public_report(report_secondary(core, free, status, targets, manifest, diagnostics))
    with args.out.open("x", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, allow_nan=False)
        fh.write("\n")
    print(json.dumps({key: report[key] for key in ("validation_status", "coverage_status",
                                                "n_selected_publications", "n_core_rows_validated")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
