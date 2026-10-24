# Prepare the selected external CRISPR screens

This directory converts user-supplied original tables into separate gene-by-publication `tested`,
`hit` and full represented `pool` masks. It does not download data or run a model. The selection and
literal source rules are fixed in the 40-row `targets.csv`. The original largest eligible raw
screen remains selected after mapping. In particular, the U937 High/SD2 table is used for
`pmid_41285767`; the smaller Low table is not substituted. Its literal union has 170 hits, compared
with 167 printed in the article. Both counts remain documented.

## Environment and shared inputs

Use Python 3.9 or newer with the packages in `requirements.txt`. The PDF reader also needs the
`pdftotext` executable from Poppler on the command path. The table reads were tested with NumPy 2.0.2,
pandas 2.3.3, openpyxl 3.1.5 and `pdftotext` 26.04.0.

Prepare the ORCS 2.0.18 human corpus and DepMap 21Q4-aligned corpus using the main package's
`build-corpus` and `build-static` commands. Their source versions and identifiers are in the main
data documentation. This preparation requires the full 81,696-gene corpus NPZ and the 17,268-gene
represented corpus NPZ, each with a `genes` array. The represented genes must be a subset of the
full corpus. This script uses gene membership, not feature values. A mean-effect vector alone is
not the represented gene definition, and target preparation does not recreate rank 64 features.

For example, with separately obtained provider files:

```bash
screenprior build-corpus --orcs /path/to/orcs-screens \
  --index /path/to/human-SCREEN_INDEX.index.tab.txt --out /path/to/corpus.npz
screenprior build-static --depmap /path/to/CRISPR_gene_effect.csv \
  --corpus /path/to/corpus.npz --out-corpus /path/to/corpus.depmap.npz \
  --features /path/to/features.npy --gene-effect /path/to/ge.npy
```

These commands describe input preparation. This directory's validation compares reconstruction
using supplied corpus inputs; it does not claim an independent fresh download of every provider
file or regeneration of the shared feature block.

## Supply the original tables

`input-files.csv` lists 45 file keys for the selected sources and one alternative full workbook.
These are filenames/aliases, not redistributed data. The default layout is:

```text
source-root/
  pmid_40615675/
    41588_2025_2222_MOESM4_ESM.xlsx
  pmid_39518122/
    cancers-3260552-supplementary.pdf
  pmid_41882347/
    pmid_41882347_MOESM3.xlsx
    pmid_41882347_MOESM6.xlsx
    pmid_41882347_MOESM7.xlsx
  ...
  pmid_41564866/
    mmc3.xlsx
  pmid_41285767/
    41467_2025_65277_MOESM3_ESM.xlsx
```

The recorded publisher/repository URL, selected sheet, input counts and call rule are in
`targets.csv`. Some files were distributed in nested supplementary bundles, and some retained
filenames are aliases. Save a provider file under the listed key or map its actual filename through
`--source-map`. Do not rename data contents or reinterpret a source criterion to make counts agree.

The path map is a local user file and is not part of a public dataset:

```json
{
  "pmid_41564866": {"mmc3.xlsx": "/licensed-files/1-s2.0-S266637912500638X-mmc3.xlsx"},
  "pmid_41882347": {
    "pmid_41882347_MOESM3.xlsx": "/licensed-files/42003_2026_9878_MOESM3_ESM.xlsx",
    "pmid_41882347_MOESM6.xlsx": "/licensed-files/42003_2026_9878_MOESM6_ESM.xlsx",
    "pmid_41882347_MOESM7.xlsx": "/licensed-files/42003_2026_9878_MOESM7_ESM.xlsx"
  }
}
```

Paths in this JSON can be absolute or relative to `--source-root`. A mistaken explicit path fails
instead of falling back to another file. Source rules do not depend on the user's directory names.

For a bundled table, supply its listed outer archive with its original nested members. Alternatively,
place the exact final workbook/text member under the publication directory. An explicit map key
such as `bundle.zip > tables.zip > table.xlsx` can point directly to that extracted member. The
literal archive-member names are retained in `readers.py`; a missing member is an error.

The large uveal-melanoma workbook can be supplied whole. The reader opens its original
`xl/sharedStrings.xml`, `xl/worksheets/sheet8.xml` and `xl/worksheets/sheet10.xml` members without
extracting them to disk. An alternative is to supply all three listed member files, with shared
strings gzip-compressed, under their recorded keys. The sparse-cell and author-hit-flag logic is
the same on both routes.

The `mmc3.xlsx` input must be the full gene-level CRC workbook. A comparison-count summary cannot
replace it. The FPR1 source requires the universe workbook plus both author hit lists. The GEO
source requires gene and sgRNA summaries. The pancreatic-screen PDF reader retains its documented
text-extraction limitation; it does not recover all transcript-level printed calls. The GitHub
source locator for `pmid_41887225` follows `main`, so an immutable upstream version is not supplied
by this directory. Original source URLs were recorded but were not fetched again for validation.

## Reconstruct masks

From the code-release root:

```bash
python -B preparation/prepare_targets.py \
  --source-root /path/to/source-root \
  --source-map /path/to/optional-path-map.json \
  --corpus /path/to/corpus.npz \
  --represented-corpus /path/to/corpus.depmap.npz \
  --out /path/to/output/targets.npz
```

Omit `--source-map` when the default layout is used. The command writes `targets.npz` and
`targets.counts.csv` only after every selected source has passed its raw-row, raw-gene, mapped-gene
and represented-gene count checks and eligibility floors. Existing output paths are refused. A
missing source, missing required column, wrong sheet, changed coverage or discrepant hit set must
be resolved against the source, not by changing the expected counts.

The NPZ fields are:

| Field | Shape and meaning |
|---|---|
| `genes` | 81,696 identifiers in the supplied full corpus order. |
| `pub_id` | 40 publication keys in `targets.csv` slot order. |
| `tested` | 81,696×40 booleans indicating a measured source gene after mapping. |
| `hit` | 81,696×40 booleans, always a subset of `tested`. |
| `pool` | 81,696×40 booleans: `tested` intersected with represented gene membership. |

The output can be projected onto the represented corpus by exact identifier indexing. Genes absent
from a source library are untested rather than measured negatives. Controls are removed before
exact mapping; documented source-specific encodings, date cells and named legacy-symbol repairs
are then applied. Scored and hit statuses collapse separately with OR on duplicate gene identities.
No fuzzy or case-insensitive matching is performed. The masks refer to the full represented pool,
before any optional essential-gene removal.

## Publication blocking configuration

`author-blocks.csv` contains40 rows, each with a target publication key and a JSON list of blocked
donor publication IDs. These are the scientific evaluation configuration. No author names, overlap
keys or row-level metadata inventory are included. Donor IDs are the corpus's original publication
identifiers and should not be relabelled.

`load_author_blocks(path, selected_ids)` in `prepare_targets.py` returns the dictionary of sets and
refuses a changed target order or duplicate donor IDs. Supply each set to the external-target
adapter's `blocked_donor_publications` argument. Source-to-gene reconstruction does not fetch author metadata or
recompute blocking decisions. Optional regeneration of those lists is a separate interface.

## Tests and rights

```bash
python -B -m pytest -q -p no:cacheprovider preparation/test_preparation.py
```

The tests cover valid archive/workbook reads, known missing files/columns, source-row misalignment,
the selected High screen, count discrepancies, duplicate/control mapping and blocking-config
coverage. They do not run models.

The processing code uses the project's software licence. Third-party tables and source-derived
arrays are not included here. Obtain and use originals under their applicable terms. Article-level
licences differ, including NC and ND statements; the code and configuration do not grant new rights
to those source materials. This directory implements source-to-gene reconstruction, not a claim
that every study dependency has already been deposited or freshly reproduced.
