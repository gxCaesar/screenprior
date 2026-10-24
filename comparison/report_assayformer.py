"""Report the matched AssayFormer comparison from saved rows, without model inference."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from adjudication import adjudicate_rows
from analysis_functions import bootstrap, validate_assayformer_rows
from row_validation import BATCH, BUDGET, STEPS, validate_route_rows
from study_metadata import audit_metadata_counts, validate_metadata

DEFAULT_DATA = Path(__file__).resolve().parents[1] / "data" / "study" / "assaybench20"


def report_saved_rows(route: pd.DataFrame, assayformer: pd.DataFrame,
                      metadata: pd.DataFrame) -> dict:
    """Validate every stored cell and reproduce the comparison's existing numerical rule.

    This validates saved counts and denominators, not raw hit labels or checkpoint inference.
    The nested subset's publication interval is labelled separately from the original outcome.
    """
    meta = validate_metadata(metadata)
    names = meta.screen.tolist()
    never = meta.loc[meta.never_read, "screen"].tolist()
    if len(route) != 200:
        raise ValueError("ScreenPrior: expected exactly 200 saved screen/seed rows")
    for column, expected in (("policy", "task-thompson"), ("condition", "P")):
        if column not in route or not route[column].eq(expected).all():
            raise ValueError(f"ScreenPrior: all saved rows must have {column}={expected}")
    tt = validate_route_rows(route, names)
    af = validate_assayformer_rows(assayformer, tt, names)
    audit_metadata_counts(tt, af, meta)
    result, differences, groups = adjudicate_rows(tt, af, names, never)
    nested = differences.reindex(never)
    nested_groups = tt.groupby("screen").pmid.first().reindex(never).astype(str).to_numpy()
    result.update({
        "analysis": "saved-row matched-pool comparison",
        "method_display": {"task-thompson": "ScreenPrior", "assayformer": "AssayFormer"},
        "model_inference_performed": False,
        "coverage": {"screens": len(names), "publication_groups": len(set(groups)),
                     "seeds": list(range(10)), "batch": BATCH, "steps": STEPS,
                     "budget": BUDGET, "screen_order": names,
                     "matched_pool_counts_agree": True, "full_library_counts_agree": True},
        "uncertainty": {
            "estimand": "mean of screen differences after averaging seeds within each screen",
            "resamples": 10000, "seed": 0,
            "screen_interval_role": "original outcome interval",
            "publication_interval_role": "cluster resampling retains the screen-mean estimand",
            "nested_never_read": {
                "screens": len(never), "publication_groups": len(set(nested_groups)),
                "ci95_publications": bootstrap(nested.to_numpy(), nested_groups),
                "role": "figure-derived interval for a nested subset, not independent confirmation",
            },
        },
    })
    return result


def load_report(screenprior_rows: Path, assayformer_rows: Path, metadata: Path) -> dict:
    route = pd.read_csv(screenprior_rows, dtype={"screen": str, "pmid": str})
    af = pd.read_csv(assayformer_rows, dtype={"screen": str})
    meta = pd.read_csv(metadata, dtype={"screen": str, "publication_group": str})
    return report_saved_rows(route, af, meta)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screenprior-rows", type=Path,
                        default=DEFAULT_DATA / "screenprior_rows.csv")
    parser.add_argument("--assayformer-rows", type=Path,
                        default=DEFAULT_DATA / "assayformer_matched_return_rows.csv")
    parser.add_argument("--metadata", type=Path, default=DEFAULT_DATA / "screens.csv")
    parser.add_argument("--out", type=Path, help="Write scientific JSON to a new file.")
    args = parser.parse_args(argv)
    try:
        result = load_report(args.screenprior_rows, args.assayformer_rows, args.metadata)
        output = json.dumps(result, indent=2, allow_nan=False) + "\n"
        if args.out is not None:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            with args.out.open("x") as stream:
                stream.write(output)
        else:
            print(output, end="")
    except (ValueError, KeyError, OSError) as error:
        parser.exit(1, f"REFUSING: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
