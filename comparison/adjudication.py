"""Apply the reported matched-pool estimand to validated saved rows."""
from __future__ import annotations

import numpy as np
import pandas as pd
from analysis_functions import bootstrap, signed_rank_p


def adjudicate_rows(tt: pd.DataFrame, af: pd.DataFrame, names: list[str],
                    never: list[str]) -> tuple[dict, pd.Series, np.ndarray]:
    """Return the reported comparison plus ordered effects for additional labelled intervals."""
    ours = tt.groupby("screen").agg(ef=("ef", "mean"),
                                    found=("found", "mean"), hits=("total_hits", "first"),
                                    pmid=("pmid", "first"))

    ours["frac"] = ours.found / ours.hits

    if ours.pmid.nunique() != 15:
        raise ValueError("the 20 comparison screens must retain their 15 publication groups")

    af["frac"] = af.found / af.total_hits

    theirs = af.groupby(["screen", "pool"])[["ef", "frac", "found"]].mean().unstack("pool")

    d = (ours.ef - theirs[("ef", "matched")]).reindex(names)

    groups = ours.pmid.reindex(names).astype(str).to_numpy()

    lo, hi = bootstrap(d.to_numpy())

    clustered = bootstrap(d.to_numpy(), groups)

    mean = float(d.mean())

    if lo > 0:
        label = "M1"
    elif hi < 0:
        label = "M2"
    else:
        label = "M3"

    frac_d = ((ours.found - theirs[("found", "matched")]) / ours.hits).reindex(names)

    by_publication = pd.Series(d.to_numpy(), index=groups).groupby(level=0).mean()

    result = {
        "D": mean, "ci95_screens": [lo, hi], "ci95_publications": clustered, "outcome": label,
        "task_thompson_ef": float(ours.ef.mean()),
        "assayformer_ef_matched": float(theirs[("ef", "matched")].mean()),
        "assayformer_ef_native": float(theirs[("ef", "native")].mean()),
        "native_to_matched_change": float((theirs[("ef", "matched")] - theirs[("ef", "native")]).mean()),
        "paired_frac_hits": {"margin": float(frac_d.mean()), "won": int((frac_d > 0).sum()),
                             "p": signed_rank_p(frac_d)},
        "paired_ef": {"won": int((d > 0).sum()), "p": signed_rank_p(d)},
        "publications_won": int((by_publication > 0).sum()), "publications": len(by_publication),
        "never_read": {"n": len(never), "D": float(d.reindex(never).mean()),
                       "won": int((d.reindex(never) > 0).sum()),
                       "p": signed_rank_p(d.reindex(never)) if len(never) > 0 else None,
                       "task_thompson_ef": float(ours.ef.reindex(never).mean()),
                       "assayformer_ef_matched":
                           float(theirs[("ef", "matched")].reindex(never).mean()),
                       "assayformer_ef_native":
                           float(theirs[("ef", "native")].reindex(never).mean()),
                       "native_to_matched_change": float(
                           (theirs[("ef", "matched")] - theirs[("ef", "native")])
                           .reindex(never).mean()),
                       "paired_frac_hits": {"margin": float(frac_d.reindex(never).mean()),
                           "won": int((frac_d.reindex(never) > 0).sum()),
                           "p": signed_rank_p(frac_d.reindex(never)) if never else None}},
        "validated_cells": {"task_thompson": len(tt), "assayformer": len(af)},
    }
    return result, d, groups
