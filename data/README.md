# Data sources and study results

The source registry in `manifests/sources.yaml` records releases, scientific identifiers,
provider URLs and citations. It contains references to data, rather than the data themselves.
Download inputs from the provider and follow its access and licence terms.

## Companion study data

The separate `screenprior-study-data-20261024.zip` archive places generated scientific results
under `data/study/` when extracted beside the code archive. It contains the saved per-cycle rows,
condition coverage and donor diagnostics for the selected 40 external publications, the saved
AssayFormer comparison endpoints, supplementary scientific tables and figure-panel values.
The [study guide](study/README.md), [data dictionary](study/DATA_DICTIONARY.md) and
[file catalogue](study/data-files.csv) define coverage, columns and evidence boundaries.

Run the primary and secondary external reporters and the saved comparison command from the main
README to recalculate these analyses. The reporters use stored generated results and do not fit
an acquisition model. Publication-level seed averaging and the original comparison denominators
are retained. The dataset does not supply third-party raw gene tables, source tested/hit masks,
the donor/static arrays or model checkpoints.

## Required inputs for the core pipeline

1. Download the human BioGRID ORCS 2.0.18 archive at the recorded provider URL. Extract it into
   its own directory. Locate the release's `SCREEN_INDEX` and the
   `BIOGRID-ORCS-SCREEN_*.screen.tab.txt` files. Use this human index with the human screen files.
2. Download DepMap 21Q4 `CRISPR_gene_effect.csv` from the recorded figshare dataset. Preserve the
   provider's filename. The file has cell lines as rows and gene labels as columns.
3. Run `build-corpus` and `build-static` as shown in the main README. The latter intersects gene
   symbols, sorts the gene order and writes the aligned corpus, representation and mean effects.

No data download occurs during installation, tests or the synthetic example. The small example
creates its own ORCS-format tables and DepMap-format CSV.

## Other recorded sources

The registry also identifies GeneDisco 1.0.5, STRING 12.0 links and aliases, NCBI human gene
information and Reactome release 97. These are source references for broader analyses. Their
benchmark-specific loaders and experiment runners are outside this minimal core package.

The NCBI URL is updated regularly and the Reactome URL follows the current release. A fresh
download can therefore differ from the dated source described in the registry. Obtain the
required historical release from the provider when an analysis needs it. Keep the release,
accession, selected files and actual preparation command with the analysis.

## Input responsibilities

The ORCS parser preserves hit calls and tested masks separately. It checks malformed rows and
unexpected hit labels in strict mode. `strict=False` is intended for inspecting a new release,
rather than producing a reported analysis.

External targets need an explicit identifier mapping, tested-gene definition, hit criterion,
candidate pool and donor-blocking rule. For the selected 40 publications,
[preparation/USAGE.md](../preparation/USAGE.md) supplies literal source readers and count checks,
and [evaluation/USAGE.md](../evaluation/USAGE.md) supplies the serial external-study runner and
reporters. The public code includes fixed donor-publication exclusion lists without an author-name
inventory. Reconstructing author overlap for a new cohort is a separate responsibility.

Obtain original tables under the source terms and supply generated, correctly aligned ORCS and
DepMap arrays before running new acquisition models. A source licence or access gap cannot be
resolved by substituting another table or an inferred threshold. The earlier development panel
exports support stored-value inspection and redrawing, rather than complete campaign replay.
The project-generated study data are licensed under CC BY 4.0, as recorded in the study guide.
Their versioned archive identifier will be added after the record exists. The code's MIT licence
and the generated-data licence do not replace third-party input terms.
