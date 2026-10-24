"""Load the fixed study configuration and align prepared masks by exact gene identity."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from screenprior.corpus import GENEDISCO_SIBLING_PMIDS, Corpus

HERE = Path(__file__).resolve().parent
TARGETS = HERE.parent / "preparation/targets.csv"
AUTHOR_BLOCKS = HERE.parent / "preparation/author-blocks.csv"
METADATA = HERE / "study-metadata.csv"
COUNTS = ("mapped_scored", "mapped_hits", "represented_scored", "represented_hits")


def report_metadata(targets_path=TARGETS, metadata_path=METADATA):
    """Adapt the public source index to the established reporting metadata schema.

    Composition is retained only on the source side of the join, as in the study report.
    Blank declared fields stay blank; the screen label is not a composition annotation.
    """
    selected = pd.read_csv(targets_path, dtype=str).fillna("")
    source = pd.read_csv(metadata_path, dtype=str).fillna("")
    required = {"slot", "pub_id", "grade", *COUNTS}
    if required - set(selected):
        raise ValueError(f"missing target configuration columns: {sorted(required - set(selected))}")
    if (len(selected) != 40 or selected.pub_id.duplicated().any()
            or selected.pub_id.eq("").any()
            or selected.slot.tolist() != [str(i) for i in range(1, 41)]):
        raise ValueError("target configuration must contain 40 distinct publications in slot order")
    required_metadata = {"pub_id", "grade", "window", "cell_line_or_type", "library_type",
                         "library_name", "phenotype"}
    if required_metadata - set(source):
        raise ValueError("study metadata is missing an explicitly declared composition field")
    if source.pub_id.tolist() != selected.pub_id.tolist():
        raise ValueError("study metadata must cover the same 40 targets in selection-slot order")
    if not selected.grade.eq(source.grade).all():
        raise ValueError("target configuration grade disagrees with study metadata")
    if "window" in selected and not selected.window.eq(source.window).all():
        raise ValueError("target configuration window disagrees with study metadata")
    selected["window"] = source.window
    for field in COUNTS:
        source[field] = selected[field]
    return selected, source


def public_report(value):
    """Remove input locations and private provenance from a report, preserving its estimates."""
    if isinstance(value, dict):
        return {key: public_report(item) for key, item in value.items()
                if key not in {"inputs", "provenance"}}
    if isinstance(value, list):
        return [public_report(item) for item in value]
    return value


def load_author_blocks(path, selected_ids):
    frame = pd.read_csv(path, dtype=str).fillna("")
    if set(frame) != {"pub_id", "blocked_donor_ids"} or frame.pub_id.tolist() != selected_ids:
        raise ValueError("author-block configuration must follow the exact selected target order")
    blocked = {}
    for row in frame.to_dict("records"):
        values = json.loads(row["blocked_donor_ids"])
        if (not isinstance(values, list) or any(not isinstance(v, str) or not v for v in values)
                or len(values) != len(set(values))):
            raise ValueError(f"{row['pub_id']}: invalid blocked donor publication list")
        blocked[row["pub_id"]] = set(values)
    return blocked


def _sorted_genes(genes):
    return (genes.ndim == 1 and genes.dtype.kind in {"U", "S"} and len(genes) > 0
            and np.all(genes != "") and np.array_equal(np.unique(genes), genes))


def _binary(array):
    return np.isin(array, (0, 1)).all()


def load_study(corpus_path, features_path, gene_effect_path, targets_npz,
               targets_csv=TARGETS, author_blocks_csv=AUTHOR_BLOCKS, metadata_csv=METADATA):
    """Validate the documented donor/static inputs and project all 40 prepared target masks.

    No model is fit. The full source masks, represented pool and exact integer counts must agree
    before a caller can dispatch any target. The membership projection does not reorder donors.
    """
    selected, source = report_metadata(targets_csv, metadata_csv)
    # This is the same metadata and eligibility validation used by the report.
    from report_secondary import target_metadata

    from screenprior.external import external_corpus

    metadata = target_metadata(selected, source)
    with np.load(corpus_path, allow_pickle=False) as archive:
        if any(not _binary(archive[key]) for key in ("tested", "hit")):
            raise ValueError("donor tested/hit masks must contain binary values")
        corpus = Corpus(*(archive[key].copy() for key in (
            "genes", "screen_id", "pmid", "cell_line", "phenotype")),
            archive["hit"].astype(bool), archive["tested"].astype(bool))
    features = np.load(features_path, allow_pickle=False)
    effect = np.load(gene_effect_path, allow_pickle=False)
    donor_ids = set(map(str, corpus.pmid))
    if (len(corpus.genes) != 17268 or len(corpus.screen_id) != 1217 or len(donor_ids) != 165
            or len(set(map(str, corpus.screen_id))) != 1217 or not _sorted_genes(corpus.genes)
            or donor_ids & GENEDISCO_SIBLING_PMIDS
            or features.shape != (17268, 64) or features.dtype != np.float32
            or effect.shape != (17268,) or not np.isfinite(features).all()
            or not np.isfinite(effect).all()):
        raise ValueError("donor/static inputs differ from the documented study shape/order/values")
    with np.load(targets_npz, allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in ("genes", "pub_id", "tested", "hit", "pool")}
    genes, ordered_ids = arrays["genes"], selected.pub_id.tolist()
    if len(genes) != 81696 or not _sorted_genes(genes):
        raise ValueError("target genes must be the sorted unique 81696-gene source universe")
    positions = np.searchsorted(genes, corpus.genes)
    if np.any(positions >= len(genes)) or not np.array_equal(genes[positions], corpus.genes):
        raise ValueError("represented genes are absent or misordered in the target source universe")
    if arrays["pub_id"].tolist() != ordered_ids:
        raise ValueError("NPZ publication order differs from the selected 40 slot order")
    if any(arrays[key].shape != (81696, 40) or not _binary(arrays[key])
           for key in ("tested", "hit", "pool")):
        raise ValueError("target tested/hit/pool masks must be binary 81696-by-40 arrays")
    tested, hit, pool = (arrays[key].astype(bool) for key in ("tested", "hit", "pool"))
    if (hit & ~tested).any():
        raise ValueError("a target hit occurs on an untested gene")
    represented = np.zeros(len(genes), dtype=bool)
    represented[positions] = True
    if not np.array_equal(pool, tested & represented[:, None]):
        raise ValueError("target pool differs from tested intersect represented genes")
    blocked = load_author_blocks(author_blocks_csv, ordered_ids)
    for column, info in enumerate(metadata):
        actual = (int(tested[:, column].sum()), int(hit[:, column].sum()),
                  int(pool[:, column].sum()), int(hit[positions, column].sum()))
        if actual != tuple(info[key] for key in COUNTS):
            raise ValueError(f"{info['pub_id']}: target mask counts disagree with selected source")
        info["pmid"] = info["pub_id"].removeprefix("pmid_")
        info["screen_id"] = f"A1-{info['pub_id']}"
        info["phenotype"] = info.get("phenotype") or info.get("screen_label", "")
        if blocked[info["pub_id"]] - donor_ids:
            raise ValueError(f"{info['pub_id']}: unknown blocked donor publications")
        # Validate all publication identities and donor exclusions before dispatching any fit.
        temporary = external_corpus(corpus, tested[positions, column], hit[positions, column],
                                    info["pub_id"], info["screen_id"], info,
                                    blocked_donor_publications=blocked[info["pub_id"]])
        del temporary
    return {"corpus": corpus, "features": features, "gene_effect": effect, "metadata": metadata,
            "tested": tested[positions], "hit": hit[positions], "blocked": blocked}
