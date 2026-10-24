"""Exact-first gene mapping and separate scored/hit mask construction.

Duplicate identities collapse with OR separately for scored and hit status. Only the
documented date, source-encoding and four named legacy-symbol repairs are permitted.
"""

import datetime as dt
import re

import numpy as np
import pandas as pd
from openpyxl.utils.datetime import from_excel

CONTROL = re.compile(r"non-?target|^ntctrl$|safe-?harbou?r|^control|^nt$|"
                     r"^(?:egfp|lacz|luciferase)$|^no-?target(?:$|[_-])|^amalgam_|^random$",
                     re.IGNORECASE)

DATE_TEXT = re.compile(r"^(\d{4})-(\d{2})-(\d{2})(?:[ T]00:00:00)?$")

LEGACY = {"TCEB1": "ELOC", "GNB2L1": "RACK1", "SRPR": "SRPRA", "TCEB2": "ELOB"}



def text_id(value):
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def documented_date(raw):
    """Reverse only the preregistered MARCH/SEPT/DEC date conversions; ignore unrelated dates."""
    match = DATE_TEXT.fullmatch(raw)
    if match:
        try:
            # A date-converted symbol encodes a calendar month/day, not a timed instant.
            value = dt.datetime(*map(int, match.groups()))  # noqa: DTZ001
        except ValueError:
            return None
    elif re.fullmatch(r"\d{5}(?:\.0+)?", raw):
        # Numeric serials are interpreted only in the specified source tables.
        try:
            value = from_excel(float(raw))
        except (OverflowError, ValueError):
            return None
        if not isinstance(value, dt.datetime):
            return None
    else:
        return None
    family = {3: "MARCH", 9: "SEPT", 12: "DEC"}.get(value.month)
    return f"{family}{value.day}" if family else None


def map_id(value, corpus_genes, pub_id):
    """Exact-first corpus mapping; no fuzzy, case-insensitive or unrestricted alias matching."""
    raw = text_id(value)
    if not raw:
        return "", "empty_id"
    if CONTROL.search(raw):
        return "", "control"
    if raw in corpus_genes:
        return raw, "exact"
    if pub_id == "pmid_42694111":
        fixed = re.sub(r"_P[12]$", "", raw)
        if fixed != raw and fixed in corpus_genes:
            return fixed, "documented_promoter_suffix"
    if pub_id == "pmid_39715471" and "__" in raw:
        fixed = raw.replace("__", "-")
        if fixed in corpus_genes:
            return fixed, "documented_double_underscore"
    numeric_date = bool(re.fullmatch(r"\d{5}(?:\.0+)?", raw))
    serial_sources = {"pmid_40615675", "pmid_39251587", "pmid_42793135"}
    fixed = (None if numeric_date and pub_id not in serial_sources
             else documented_date(raw))
    if fixed and fixed in corpus_genes:
        return fixed, "documented_excel_date"
    fixed = LEGACY.get(raw)
    if fixed in corpus_genes:
        return fixed, "documented_legacy"
    return "", "unmapped"


def eligibility(mapped_scored, mapped_hits, represented_scored, represented_hits):
    """Mapped corpus eligibility and the frozen full-pool RunConfig are separate gates."""
    reasons = []
    if mapped_scored < 12000:
        reasons.append("mapped_scored<12000")
    if mapped_hits < 30:
        reasons.append("mapped_hits<30")
    if mapped_scored and mapped_hits / mapped_scored > .20:
        reasons.append("mapped_hit_rate>0.20")
    if represented_scored < 5000:
        reasons.append("represented_scored<5000")
    if represented_hits < 30:
        reasons.append("represented_hits<30")
    return reasons


def assemble_rows(raw, corpus_genes, static_genes, pub_id):
    """Map rows, retain the trace, and OR scored/hit status on the mapped gene identity."""
    corpus_set = set(corpus_genes)
    trace = []
    for ordinal, row in enumerate(raw.itertuples(index=False), start=1):
        source = text_id(row.raw_id)
        mapped, rule = map_id(source, corpus_set, pub_id)
        trace.append({"source_row": ordinal, "raw_id": source, "mapped_gene": mapped,
                      "mapping_rule": rule, "scored": bool(row.scored), "hit": bool(row.hit),
                      "represented": bool(mapped and mapped in static_genes)})
    frame = pd.DataFrame(trace)
    valid_raw = frame.scored & ~frame.mapping_rule.isin(["control", "empty_id"])
    raw_scored = frame.loc[valid_raw, "raw_id"].nunique()
    raw_hits = frame.loc[valid_raw & frame.hit, "raw_id"].nunique()
    mapped = frame[frame.mapped_gene != ""]
    collapsed = mapped.groupby("mapped_gene", sort=True)[["scored", "hit"]].any()
    if (collapsed.hit & ~collapsed.scored).any():
        raise ValueError(f"{pub_id}: mapped hit lacks a measured score")
    scored_genes = set(collapsed.index[collapsed.scored])
    hit_genes = set(collapsed.index[collapsed.hit])
    represented = scored_genes & static_genes
    represented_hits = hit_genes & represented
    n_scored, n_hits = len(scored_genes), len(hit_genes)
    reasons = eligibility(n_scored, n_hits, len(represented), len(represented_hits))
    matched_ids = frame.loc[valid_raw & (frame.mapped_gene != ""), "raw_id"].nunique()
    counts = {"raw_rows": len(raw), "raw_scored": int(raw_scored), "raw_hits": int(raw_hits),
              "mapped_scored": n_scored, "mapped_hits": n_hits,
              "hit_rate": n_hits / n_scored if n_scored else 0,
              "raw_ids_mapped": int(matched_ids),
              "in_corpus_fraction": matched_ids / raw_scored if raw_scored else 0,
              "represented_scored": len(represented), "represented_hits": len(represented_hits),
              "represented_hit_rate": (len(represented_hits) / len(represented)
                                       if represented else 0),
              "duplicate_ids": int(frame.loc[valid_raw, "raw_id"].duplicated().sum()),
              "mapped_duplicate_rows": int(mapped.loc[mapped.scored, "mapped_gene"]
                                           .duplicated().sum()),
              "date_ids": int(frame.loc[frame.mapping_rule == "documented_excel_date", "raw_id"]
                              .nunique()),
              "control_ids": int(frame.loc[frame.mapping_rule == "control", "raw_id"].nunique()),
              "unmapped_ids": int(frame.loc[valid_raw & (frame.mapping_rule == "unmapped"),
                                            "raw_id"]
                                  .nunique())}
    arrays = {"tested": np.isin(corpus_genes, list(scored_genes)),
              "hit": np.isin(corpus_genes, list(hit_genes)),
              "pool": np.isin(corpus_genes, list(represented))}
    return counts, arrays, trace, reasons

