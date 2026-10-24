"""Validate complete primary rows and apply the publication-level outcome rule.

Seeds are averaged within publications. Exact acquired-hit counts define every contrast;
incomplete grids, changed denominators and missing seeds are refused.
"""

# Keep the established validation API and dictionary construction.
# ruff: noqa: C408, TRY004

from __future__ import annotations

import argparse
import json
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from screenprior.metrics import signed_rank_p as _signed_rank_p

CORE_POLICIES = (
    "random", "greedy", "topuncertain", "softuncertain", "coreset", "badge",
    "task-thompson", "task-thompson/permuted", "task-thompson/uniform",
)
SEEDS, CYCLES = tuple(range(10)), tuple(range(1, 11))
BATCH, BUDGET = 64, 640
COUNTS = ("seed", "cycle", "n_genes", "n_hits", "n_donors", "acquired", "acquired_hits")
IDENTITIES = ("pub_id", "screen_id", "pmid", "policy", "pool")


def _boolean(value, expected):
    if isinstance(value, (bool, np.bool_)):
        return bool(value) is expected
    if isinstance(value, str):
        return value.strip().casefold() == str(expected).casefold()
    return False


def _expected_ids(expected_pub_ids):
    if isinstance(expected_pub_ids, str):
        raise ValueError("expected publication identifiers must be a collection, not one string")
    values = list(expected_pub_ids)
    if (not values or any(not isinstance(p, str) or not p or p != p.strip() for p in values)
            or len(set(values)) != len(values)):
        raise ValueError("expected publication identifiers must be nonempty, distinct strings")
    return sorted(values)


def strict_validate_primary(rows, expected_pub_ids):
    """Return an unfiltered normalized frame after verifying every declared primary cell.

    Row dictionaries and pandas DataFrames are accepted. Decimal CSV round-trip differences below
    1e-12 are tolerated in the two printed ratios only; inference uses the exact integer hit
    counts. No tolerance applies to identities, count constancy, the budget or missing cells.
    """
    expected = _expected_ids(expected_pub_ids)
    frame = pd.DataFrame(rows).copy(deep=True)
    required = set(IDENTITIES + COUNTS + ("recall", "chance", "author_blocking", "block_cell_line"))
    if required - set(frame.columns):
        raise ValueError(f"missing primary columns: {sorted(required - set(frame.columns))}")
    for column in IDENTITIES:
        if frame[column].isna().any():
            raise ValueError(f"missing primary identity: {column}")
        frame[column] = frame[column].map(str)
        if frame[column].map(lambda s: not s or s != s.strip()).any():
            raise ValueError(f"empty or untrimmed primary identity: {column}")
    observed = set(frame.pub_id)
    if observed != set(expected):
        raise ValueError(f"publication mismatch: missing={sorted(set(expected) - observed)}, "
                         f"extra={sorted(observed - set(expected))}")
    if not frame.pool.eq("full").all():
        raise ValueError("primary rows must use pool=full")
    if not frame.author_blocking.map(lambda v: _boolean(v, True)).all():
        raise ValueError("primary rows must explicitly have author_blocking=True")
    if not frame.block_cell_line.map(lambda v: _boolean(v, False)).all():
        raise ValueError("primary rows must retain block_cell_line=False")
    frame["author_blocking"], frame["block_cell_line"] = True, False
    for column in COUNTS + ("recall", "chance"):
        try:
            frame[column] = pd.to_numeric(frame[column], errors="raise")
        except (ValueError, TypeError) as exc:
            raise ValueError(f"primary {column} must be numeric") from exc
        if not np.isfinite(frame[column].to_numpy(dtype=float)).all():
            raise ValueError(f"primary {column} must be finite")
    for column in COUNTS:
        values = frame[column].to_numpy()
        if (values != np.floor(values)).any() or (np.abs(values) >= 2 ** 63).any():
            raise ValueError(f"primary {column} must contain finite integer counts")
        frame[column] = frame[column].astype(np.int64)
    keys = ["pub_id", "seed", "policy", "cycle"]
    if frame.duplicated(keys).any():
        examples = frame.loc[frame.duplicated(keys, keep=False), keys].head(4).to_dict("records")
        raise ValueError(f"duplicate primary cells: {examples}")
    actual_keys = set(frame[keys].itertuples(index=False, name=None))
    expected_keys = set(product(expected, SEEDS, CORE_POLICIES, CYCLES))
    if actual_keys != expected_keys:
        missing = sorted(expected_keys - actual_keys)
        extra = sorted(actual_keys - expected_keys)
        raise ValueError(f"primary cell coverage mismatch: {len(missing)} missing, "
                         f"{len(extra)} extra; examples={missing[:2] or extra[:2]}")
    for column in ("n_genes", "n_hits", "n_donors", "screen_id", "pmid"):
        if frame.groupby("pub_id", sort=False)[column].nunique().ne(1).any():
            raise ValueError(f"primary {column} changes within a publication")
    publication_pmids = frame.groupby("pub_id", sort=False).pmid.first()
    if publication_pmids.duplicated().any():
        raise ValueError("different pub_id values cannot count the same PMID as independent units")
    if ((frame.n_genes < 5000).any() or (frame.n_hits < 30).any() or
            (frame.n_hits > frame.n_genes).any() or (frame.n_donors < 1).any()):
        raise ValueError("primary N/H/donor counts violate the frozen eligibility")
    if frame.acquired.ne(BATCH * frame.cycle).any():
        raise ValueError("primary acquired count must equal 64 times cycle")
    if ((frame.acquired > frame.n_genes).any() or (frame.acquired_hits < 0).any() or
            (frame.acquired_hits > np.minimum(frame.acquired, frame.n_hits)).any()):
        raise ValueError("primary acquired-hit count exceeds its acquired/H bound")
    if not np.allclose(frame.recall, frame.acquired_hits / frame.n_hits,
                       rtol=0, atol=1e-12):
        raise ValueError("primary recall does not equal acquired_hits/H")
    if not np.allclose(frame.chance, frame.acquired / frame.n_genes, rtol=0, atol=1e-12):
        raise ValueError("primary chance does not equal acquired/N")
    frame = frame.sort_values(keys, kind="stable").reset_index(drop=True)
    changes = frame.groupby(["pub_id", "seed", "policy"], sort=False).acquired_hits.diff()
    if ((changes < 0) | (changes > BATCH)).any():
        raise ValueError("primary acquired hits must be monotonic with at most 64 new hits/cycle")
    return frame


def signed_rank_p(values):
    """The registered routine, with the undefined all-zero call represented by null P=1."""
    differences = np.asarray(values, dtype=float)
    if len(differences) and np.all(differences == 0):
        return 1.0
    return _signed_rank_p(differences)


def adjudicate_primary(rows, expected_pub_ids):
    """Apply the mutually exclusive amendment-1 outcome rule to validated publication means."""
    expected = _expected_ids(expected_pub_ids)
    frame = strict_validate_primary(rows, expected)
    final = frame.loc[frame.cycle.eq(10)]
    publications = []
    for pub_id in expected:
        group = final.loc[final.pub_id.eq(pub_id)]
        identity = group.iloc[0]
        arms = {name: group.loc[group.policy.eq(name)].sort_values("seed", kind="stable")
                for name in CORE_POLICIES}
        # Exact counts first: equal acquisitions must yield an exactly zero paired contrast.
        ours_sum = int(arms["task-thompson"].acquired_hits.sum())
        badge_sum = int(arms["badge"].acquired_hits.sum())
        hit_difference = (ours_sum - badge_sum) / len(SEEDS)
        hits = int(identity.n_hits)
        recalls = {name: int(arm.acquired_hits.sum()) / (len(SEEDS) * hits)
                   for name, arm in arms.items()}
        publications.append(dict(
            pub_id=pub_id, pmid=str(identity.pmid), screen_id=str(identity.screen_id),
            n_genes=int(identity.n_genes), n_hits=hits, n_donors=int(identity.n_donors),
            n_seeds=len(SEEDS), acquired=BUDGET, chance=BUDGET / int(identity.n_genes),
            screenprior_recall=recalls["task-thompson"], badge_recall=recalls["badge"],
            screenprior_acquired_hits=ours_sum / len(SEEDS),
            badge_acquired_hits=badge_sum / len(SEEDS),
            margin=(ours_sum - badge_sum) / (len(SEEDS) * hits),
            additional_hits=hit_difference, policy_mean_recall=recalls,
        ))
    differences = np.array([r["margin"] for r in publications], dtype=float)
    additional_hits = np.array([r["additional_hits"] for r in publications], dtype=float)
    rng = np.random.default_rng(0)
    boot = differences[rng.integers(len(publications), size=(10000, len(publications)))].mean(1)
    lower, upper = np.percentile(boot, [2.5, 97.5])
    mean = float(differences.mean())
    p = signed_rank_p(differences)
    if mean > 0 and p < 0.05 and lower > 0:
        outcome = "confirmed"
    elif upper < 0:
        outcome = "negative"
    else:
        outcome = "unconfirmed"
    wins, losses = int((differences > 0).sum()), int((differences < 0).sum())
    non_tied = wins + losses
    sign_p = (float(stats.binomtest(wins, non_tied, p=0.5, alternative="two-sided").pvalue)
              if non_tied else 1.0)
    return dict(
        validation_status="passed", outcome=outcome,
        scope="registered full-pool author-blocked primary ScreenPrior-minus-BADGE contrast",
        endpoint="full-pool recall@640", policy="task-thompson", comparator="badge",
        independent_unit="publication", n_publications=len(publications),
        n_rows_validated=len(frame), n_final_rows=len(final), seeds=list(SEEDS),
        cycles=list(CYCLES), core_policy_order=list(CORE_POLICIES), budget=BUDGET,
        mean_screenprior_recall=float(np.mean([r["screenprior_recall"] for r in publications])),
        mean_badge_recall=float(np.mean([r["badge_recall"] for r in publications])),
        mean_margin=mean, median_margin=float(np.median(differences)),
        margin_ci95=[float(lower), float(upper)], signed_rank_p=p,
        realized_margin_sd=float(differences.std(ddof=1)) if len(differences) > 1 else None,
        win_fraction=wins / len(publications), n_wins=wins, n_losses=losses,
        n_ties=len(publications) - non_tied, sign_test_n=non_tied, sign_binomial_p=sign_p,
        sign_test_tie_rule=("zero publication margins excluded from sign test; "
                            "win fraction uses all"),
        mean_additional_hits=float(additional_hits.mean()),
        median_additional_hits=float(np.median(additional_hits)),
        bootstrap=dict(draws=10000, seed=0, unit="publication", order="pub_id stable ascending",
                       interval="percentile 2.5 and 97.5"),
        outcome_rule=("confirmed iff mean>0 and P<0.05 and lower>0; "
                      "negative iff upper<0; else unconfirmed"),
        publications=publications,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--targets", type=Path,
                        default=Path(__file__).resolve().parent.parent / "preparation/targets.csv")
    parser.add_argument("--out", type=Path, required=True, help="fresh JSON path")
    args = parser.parse_args(argv)
    targets = pd.read_csv(args.targets, dtype={"pub_id": str})
    if "pub_id" not in targets or len(targets) != 40:
        raise ValueError("primary CLI requires the selected 40 target publication rows")
    rows = pd.read_csv(args.rows, dtype={key: str for key in IDENTITIES})
    report = adjudicate_primary(rows, targets.pub_id.tolist())
    with args.out.open("x", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, allow_nan=False)
        fh.write("\n")
    print(json.dumps({key: report[key] for key in ("validation_status", "outcome", "n_publications",
                                                "mean_margin", "margin_ci95", "signed_rank_p")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
