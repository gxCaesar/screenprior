"""Run the fixed 40-publication external study serially into a new output directory.

The public external-target API owns acquisition, surrogate fitting, donor embedding and controls.
This adapter supplies selected masks and explicit donor exclusions, then retains integer results
and acquisition traces for the primary and descriptive secondary reporters.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from adjudicate_a1 import CORE_POLICIES, strict_validate_primary
from report_secondary import _validate_diagnostics, _validate_free_grid
from study_data import AUTHOR_BLOCKS, METADATA, TARGETS, load_study

from screenprior import external

CONDITIONS = (("primary", "full", True), ("non-essential", "non-essential", True),
              ("author-blocking-off", "full", False))


def jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [jsonable(item) for item in value]
    return value


def write_csv(path, rows):
    with Path(path).open("x", newline="", encoding="utf-8") as fh:
        pd.DataFrame(rows).to_csv(fh, index=False)


def write_vectors(path, rows):
    with Path(path).open("x", encoding="utf-8") as fh:
        fh.writelines(json.dumps(jsonable(row), allow_nan=False) + "\n" for row in rows)


def condition_statuses(payload):
    statuses = []
    for index, info in enumerate(payload["metadata"]):
        pub = info["pub_id"]
        for condition, pool, blocking in CONDITIONS:
            candidate = payload["tested"][:, index].copy()
            if pool == "non-essential":
                candidate &= payload["gene_effect"] >= -0.5
            n, h = int(candidate.sum()), int(payload["hit"][candidate, index].sum())
            blocked = payload["blocked"][pub] if blocking else set()
            donors = int((~np.isin(payload["corpus"].pmid, list(blocked))).sum())
            statuses.append({"pub_id": pub, "condition": condition, "pool": pool,
                             "author_blocking": blocking, "n_genes": n, "n_hits": h,
                             "n_donors": donors, "status": "not_run", "reason": ""})
    return statuses


def validate_result(core, free, status):
    """Apply the same count/grid/posterior checks consumed by the saved-row reporters."""
    if core["status"] != "complete" or free["status"] != "complete":
        raise ValueError("external API declined an eligible target condition")
    pub = status["pub_id"]
    fields = ("pub_id", "condition", "pool", "author_blocking", "n_genes", "n_hits", "n_donors")
    if any(any(row[key] != status[key] for key in fields)
           for result in (core, free) for row in result["rows"]):
        raise ValueError("result identity/denominators differ from the prepared target condition")
    normalized = pd.DataFrame(core["rows"]).copy()
    normalized["pool"], normalized["author_blocking"] = "full", True
    strict_validate_primary(normalized, [pub])
    _validate_free_grid(pd.DataFrame(free["rows"]), [pub])
    diagnostics = pd.DataFrame(core["diagnostics"]).copy()
    normalized["condition"] = "primary"
    diagnostics["condition"], diagnostics["pool"] = "primary", "full"
    diagnostics["author_blocking"] = True
    _validate_diagnostics(diagnostics, normalized)
    if len(core["traces"]) != 90 or len(free["traces"]) != 22:
        raise ValueError("external API must retain 90 core and 22 free-control traces")


def execute(payload, out):
    """Execute in slot/condition order; a failure leaves explicitly partial aggregates."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    statuses = condition_statuses(payload)
    core_rows, free_rows = [], []
    failure = None
    for index, info in enumerate(payload["metadata"]):
        pub = info["pub_id"]
        for offset, (condition, pool, blocking) in enumerate(CONDITIONS):
            status = statuses[index * len(CONDITIONS) + offset]
            if failure is not None:
                continue
            destination = out / "targets" / f"{index + 1:03d}-{pub}" / condition
            destination.mkdir(parents=True, exist_ok=False)
            if status["n_genes"] < 5000 or status["n_hits"] < 30:
                status.update(status="not_eligible", reason="RunConfig N/H floor")
                continue
            try:
                blocked = payload["blocked"][pub] if blocking else set()
                corpus = external.external_corpus(
                    payload["corpus"], payload["tested"][:, index], payload["hit"][:, index],
                    pub, info["screen_id"], info, blocked_donor_publications=blocked)
                target = len(corpus.screen_id) - 1
                if len(corpus.donors_for(target)) != status["n_donors"]:
                    raise ValueError("donor count differs from the explicit publication exclusions")
                core = external.run_one_target(corpus, target, payload["features"],
                                               payload["gene_effect"], pub_id=pub, pool=pool)
                free = external.label_free_baselines(corpus, target, payload["gene_effect"],
                                                    pub_id=pub, pool=pool)
                for result in (core, free):
                    for kind in ("rows", "traces", "diagnostics"):
                        for row in result[kind]:
                            row.update(condition=condition, author_blocking=blocking,
                                       window=info["window"], grade=info["grade"])
                validate_result(core, free, status)
                write_csv(destination / "core_rows.csv", core["rows"])
                write_csv(destination / "free_controls.csv", free["rows"])
                write_vectors(destination / "core_traces.jsonl", core["traces"])
                write_vectors(destination / "free_control_traces.jsonl", free["traces"])
                write_vectors(destination / "after_cycle3.jsonl", core["diagnostics"])
                core_rows.extend(core["rows"])
                free_rows.extend(free["rows"])
                status["status"] = "completed"
                print(f"{index + 1:02d} {pub} {condition}: completed "
                      f"N={status['n_genes']} H={status['n_hits']} "
                      f"donors={status['n_donors']}", flush=True)
            except Exception as exc:  # noqa: BLE001 - retain partial counts before propagating
                status.update(status="failed", reason=str(exc))
                failure = exc
    write_csv(out / "condition_status.csv", statuses)
    if core_rows:
        write_csv(out / "all_core_rows.csv", core_rows)
        write_csv(out / "all_free_controls.csv", free_rows)
        primary = [row for row in core_rows if row["condition"] == "primary"]
        try:
            strict_validate_primary(primary, [row["pub_id"] for row in payload["metadata"]])
            primary_name = "primary_rows.csv"
        except ValueError:
            primary_name = "primary_rows.partial.csv"
        if primary:
            write_csv(out / primary_name, primary)
    if failure is not None:
        raise RuntimeError("study stopped after a failed condition; inspect condition_status.csv") \
            from failure
    return {"status": "completed", "publications": len(payload["metadata"]),
            "completed_conditions": sum(row["status"] == "completed" for row in statuses),
            "ineligible_conditions": sum(row["status"] == "not_eligible" for row in statuses),
            "core_rows": len(core_rows), "free_rows": len(free_rows)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("corpus", "features", "gene-effect", "targets-npz"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--targets", type=Path, default=TARGETS)
    parser.add_argument("--author-blocks", type=Path, default=AUTHOR_BLOCKS)
    parser.add_argument("--metadata", type=Path, default=METADATA)
    parser.add_argument("--out", type=Path, help="a directory that does not already exist")
    parser.add_argument("--check-inputs", action="store_true",
                        help="validate/project inputs and report eligibility without fitting")
    args = parser.parse_args(argv)
    if not args.check_inputs and args.out is None:
        parser.error("--out is required unless --check-inputs is used")
    if args.out is not None and args.out.exists():
        raise FileExistsError("output already exists; choose a fresh output directory")
    payload = load_study(args.corpus, args.features, args.gene_effect, args.targets_npz,
                         args.targets, args.author_blocks, args.metadata)
    if args.check_inputs:
        result = {"status": "passed", "model_execution": "not_run", "publications": 40,
                  "donor_screens": len(payload["corpus"].screen_id),
                  "donor_publications": len(set(payload["corpus"].pmid)),
                  "feature_shape": list(payload["features"].shape),
                  "conditions": condition_statuses(payload),
                  "core_policy_order": list(CORE_POLICIES)}
    else:
        result = execute(payload, args.out)
    print(json.dumps(result, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
