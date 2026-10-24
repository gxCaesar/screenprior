"""Reconstruct the selected 40 external-screen masks from user-supplied licensed source files."""

import argparse
import csv
import json
import tempfile
from pathlib import Path

import numpy as np
from gene_mapping import assemble_rows
from readers import read_screen
from source_files import Assets

HERE = Path(__file__).resolve().parent
COUNT_FIELDS = ("raw_rows", "raw_scored", "raw_hits", "mapped_scored", "mapped_hits",
                "represented_scored", "represented_hits")


def load_config(path):
    with open(path, newline="") as fh:
        records = list(csv.DictReader(fh))
    if len(records) != 40 or [int(r["slot"]) for r in records] != list(range(1, 41)):
        raise ValueError("target configuration must contain exactly 40 rows in slot order 1..40")
    if len({r["pub_id"] for r in records}) != 40:
        raise ValueError("target configuration contains duplicate publications")
    for row in records:
        for field in COUNT_FIELDS:
            if int(row[field]) <= 0:
                raise ValueError(f"{row['pub_id']}: missing/nonpositive expected {field}")
        if not row["hit_criterion"]:
            raise ValueError(f"{row['pub_id']}: missing selected author-call rule")
        if row["pub_id"] == "pmid_41285767" and (
            row["sheet"] != "SD2" or row["screen_label"] != "U937 IL-1beta-high vs pre-sort"
            or int(row["raw_scored"]) != 19086 or int(row["raw_hits"]) != 170
        ):
            raise ValueError("pmid_41285767 must use the selected High SD2 table and literal 170")
    return records


def load_genes(path):
    with np.load(path, allow_pickle=False) as archive:
        genes = archive["genes"].copy()
    if genes.ndim != 1 or len(genes) == 0 or len(np.unique(genes)) != len(genes):
        raise ValueError("corpus genes must be a nonempty one-dimensional unique sequence")
    if genes.dtype.kind not in {"U", "S"} or np.any(genes == ""):
        raise ValueError("corpus gene identifiers must be nonempty strings")
    return genes


def load_author_blocks(path, selected_ids):
    """Read the supplied publication-ID lists without reconstructing author identities."""
    with open(path, newline="") as fh:
        records = list(csv.DictReader(fh))
    if len(records) != 40 or [row["pub_id"] for row in records] != list(selected_ids):
        raise ValueError("author-block configuration differs from the selected target order")
    output = {}
    for row in records:
        blocked = json.loads(row["blocked_donor_ids"])
        if (not isinstance(blocked, list) or any(not isinstance(x, str) or not x for x in blocked)
                or len(set(blocked)) != len(blocked)):
            raise ValueError(f"{row['pub_id']}: invalid or duplicate blocked donor IDs")
        output[row["pub_id"]] = set(blocked)
    return output


def check_counts(selected, actual, reasons):
    errors = [f"{field}: expected {selected[field]}, observed {actual[field]}"
              for field in COUNT_FIELDS if int(selected[field]) != int(actual[field])]
    if errors or reasons:
        raise ValueError(f"{selected['pub_id']}: source/mapping coverage differs; "
                         + "; ".join(errors + reasons))


def prepare(config, source_root, corpus, represented_corpus, out, source_map=None):
    selected = load_config(config)
    genes, represented = load_genes(corpus), load_genes(represented_corpus)
    if len(genes) != 81696 or len(represented) != 17268:
        raise ValueError("this configuration requires the 81696-gene corpus and 17268-gene "
                         "represented corpus from the documented source releases")
    if not set(represented.tolist()) <= set(genes.tolist()):
        raise ValueError("represented corpus contains genes outside the full corpus")
    out = Path(out)
    if out.suffix != ".npz":
        raise ValueError("output path must end in .npz")
    counts_path = out.with_suffix(".counts.csv")
    if out.exists() or counts_path.exists():
        raise FileExistsError("output already exists; choose a new output path")
    assets = Assets(source_root, source_map)
    represented_set = set(represented.tolist())
    counts, vectors = [], []
    with tempfile.TemporaryDirectory(prefix="screen-tables-") as work:
        for row in selected:
            try:
                source = read_screen(row["pub_id"], row, assets, Path(work))
                measured, masks, _, reasons = assemble_rows(
                    source.rows, genes, represented_set, row["pub_id"])
                check_counts(row, measured, reasons)
            except (ValueError, KeyError, FileNotFoundError) as exc:
                raise ValueError(f"{row['pub_id']}: cannot reconstruct the selected source: "
                                 f"{exc}") from exc
            counts.append({"slot": int(row["slot"]), "pub_id": row["pub_id"],
                           **{field: measured[field] for field in COUNT_FIELDS}})
            vectors.append(masks)
            print(f"{row['slot']:>2s} {row['pub_id']} "
                  f"{measured['mapped_scored']}/{measured['mapped_hits']} -> "
                  f"{measured['represented_scored']}/{measured['represented_hits']}", flush=True)
    payload = {"genes": genes, "pub_id": np.array([row["pub_id"] for row in selected])}
    for name in ("tested", "hit", "pool"):
        payload[name] = np.column_stack([vector[name] for vector in vectors])
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **payload)
    with open(counts_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(counts[0]))
        writer.writeheader()
        writer.writerows(counts)
    return {"targets": len(selected), "genes": len(genes),
            "represented_genes": len(represented), "array_shape": list(payload["tested"].shape)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=HERE / "targets.csv")
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--source-map", type=Path,
                        help="optional JSON publication/asset-to-file mapping")
    parser.add_argument("--corpus", type=Path, required=True,
                        help="full ORCS-derived corpus NPZ with genes")
    parser.add_argument("--represented-corpus", type=Path, required=True,
                        help="DepMap-aligned corpus NPZ with genes")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.config, args.source_root, args.corpus,
                             args.represented_corpus, args.out, args.source_map), indent=2))


if __name__ == "__main__":
    main()
