# Reproduce the external 40-publication study

This directory connects the selected source tables to the public `screenprior.external` API and
the publication-level reports. It covers the external study. The earlier 153-publication
development tables and their producing scripts are not included here, so these commands do not
reproduce every manuscript result.

## Dependencies and inputs

Run commands from the code-release root, using Python 3.9 or newer. Install the package and the
CPU dependency versions in the root `requirements-lock.txt`:

```bash
python -m pip install -r requirements-lock.txt
python -m pip install -e .
```

The saved-row reports were checked with NumPy 2.0.2, pandas 2.3.3, SciPy 1.13.1 and scikit-learn
1.6.1. The source readers have additional workbook/PDF requirements described in
[`preparation/USAGE.md`](../preparation/USAGE.md).

Supply the following files:

| Input | Meaning |
|---|---|
| `corpus.depmap.npz` | The documented ORCS/DepMap-aligned donor corpus, with 17,268 sorted unique genes, 1,217 screens and 165 publications, retaining the fixed GeneDisco sibling exclusions. |
| `features.npy` | Finite float32 static features, shape 17,268×64, in that corpus's gene order. |
| `ge.npy` | Finite mean DepMap gene effects, shape 17,268, in the same order. |
| `targets.npz` | The output of `preparation/prepare_targets.py`: the full 81,696-gene `tested`, `hit` and represented `pool` masks for the selected 40 targets. |
| `preparation/targets.csv` | The selected tables, literal author criteria and expected source/mapping counts, in selection-slot order. |
| `preparation/author-blocks.csv` | Each target's fixed blocked donor-publication IDs. It contains no author-name inventory. |
| `evaluation/study-metadata.csv` | The selected targets' grade, window and originally declared composition fields. Blank phenotype values stay unknown in the report. |

Generate the donor/static inputs with the package's documented corpus commands, and obtain
third-party originals under their applicable terms. Reconstruct `targets.npz` with the
preparation instructions before running this adapter. Its source/count checks retain the High
SD2 table for `pmid_41285767` and its literal 170-hit union, with the article's printed 167 recorded
as a source discrepancy. The smaller Low table is not an alternative target.

An array's shape cannot establish its scientific identity. Inputs must come from the stated source
releases and construction steps. The adapter checks exact gene membership/order, all mask counts,
the 40 target identities, eligibility and explicit donor exclusions. It cannot certify arbitrary
replacement feature values or a changed donor corpus solely from matching dimensions.

## Check inputs without fitting

```bash
python -B evaluation/run_study.py \
  --corpus /path/to/corpus.depmap.npz \
  --features /path/to/features.npy \
  --gene-effect /path/to/ge.npy \
  --targets-npz /path/to/targets.npz \
  --check-inputs
```

The command prints projected pool counts and donor counts for all 120 target-condition pairs.
It refuses mismatched source counts, reversed target order, hits on untested genes, an incorrect
represented pool, unknown donor exclusions and inputs outside the documented study dimensions.
It does not fit or score an acquisition model.

## Run the fixed study

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
python -B evaluation/run_study.py \
  --corpus /path/to/corpus.depmap.npz \
  --features /path/to/features.npy \
  --gene-effect /path/to/ge.npy \
  --targets-npz /path/to/targets.npz \
  --out /path/to/new-study-output
```

The runner is serial and requires a new output directory. It uses the supplied static features
and the unchanged public acquisition implementation. Corpus embeddings are built at rank 64
after donor blocking. Every eligible target-condition uses ten seeds, nine stable arm slots and
ten batches of 64 genes, ending at 640 acquisitions. The arm order is:

```text
random, greedy, topuncertain, softuncertain, coreset, badge,
task-thompson, task-thompson/permuted, task-thompson/uniform
```

`random` uses the continuous generator stream, including the shared opening. The full pool with
author-group blocking is primary. The two descriptive sensitivities use the non-essential pool
with author-group blocking, and the full pool with author-group blocking switched off. Mandatory
publication exclusion and fixed GeneDisco exclusions remain in every condition. Cell-line
blocking stays off. The non-essential pool retains genes with mean effect ≥ −0.5; the unchanged
eligibility floors are 5,000 genes and 30 hits. Ineligible sensitivity pools are recorded explicitly.

Two label-free scores, corpus hit frequency and negative DepMap mean effect, are each evaluated
once by pure ranking and once after each seed's shared opening. These yield four control labels
and 22 observations per eligible condition. The pure controls have seed −1 and cycle 0, with no
artificial ten-fold replication. Acquired-hit integers and pool denominators accompany all printed
ratios. After-cycle-3 donor diagnostics are retained after 192 ScreenPrior acquisitions.

The output contains:

```text
new-study-output/
  primary_rows.csv
  all_core_rows.csv
  all_free_controls.csv
  condition_status.csv
  targets/001-pmid_.../primary/
    core_rows.csv
    free_controls.csv
    core_traces.jsonl
    free_control_traces.jsonl
    after_cycle3.jsonl
  ...
```

Each completed condition has 900 core rows and 22 free-control rows. The primary table has
36,000 rows when all 40 publications finish. A failed condition stops further dispatch, preserves
explicit failed/not-run statuses and writes available aggregates. Incomplete primary coverage is
named `primary_rows.partial.csv`; it is never accepted as a complete primary by the reporter.
The serial runner has no resume or multiprocessing layer. Use a new destination for another run.

## Produce primary and secondary reports

```bash
python -B evaluation/adjudicate_a1.py \
  --rows /path/to/new-study-output/primary_rows.csv \
  --out /path/to/primary-adjudication.json

python -B evaluation/report_secondary.py \
  --core /path/to/new-study-output/all_core_rows.csv \
  --free /path/to/new-study-output/all_free_controls.csv \
  --status /path/to/new-study-output/condition_status.csv \
  --diagnostics-root /path/to/new-study-output/targets \
  --out /path/to/secondary-report.json
```

Both JSON destinations must be new. The primary reporter requires exactly the selected 40 target
IDs and the complete publication×seed×arm×cycle grid. Counts define inference; printed ratios are
checked against them. Seeds are averaged within each publication before the ScreenPrior-minus-
BADGE contrast. The 10,000-draw publication bootstrap uses seed 0 and a percentile interval.
The primary rule is confirmed only when the mean margin is positive, the two-sided signed-rank
P value is below 0.05 and the lower interval endpoint is positive. It is negative when the upper
endpoint is negative, and otherwise unconfirmed.

The secondary reporter validates condition omissions, all control observations and the primary
ten-seed donor diagnostics. It reports the registered comparator contrasts, strict-window,
grade-A and minimal-signal subsets, author-blocking-off sensitivity, donor-concentration terciles
and explicit composition summaries. Publication means precede every contrast and tercile. These
estimates are descriptive, uncorrected for multiplicity and have no secondary success decision.
Input paths and private composition provenance are omitted from the public JSON schema.

The reporters can also consume saved study rows without rerunning any acquisition model. The
scientific function bodies match the study reporters. Acceptance against saved rows found exact
equality for all 882 numeric primary fields and all 31,327 numeric secondary fields after removal
of input locations and private provenance. Coverage was 40 targets, 105,300 core rows, 2,574 free
rows and 400 primary diagnostics; three non-essential pools were ineligible. Removing one actual
publication's seed 9 removed 90 cells and was refused by both reporters. These are saved-row
reporting and input-adapter checks, not a fresh 40-publication model rerun.

## Scope of reproduction

The source-to-report chain is licensed originals → reviewed source readers → full tested/hit/pool
masks → exact represented-gene projection and donor exclusions → public external API → saved
integer rows/diagnostics → publication-level reports. This directory supplies the adapter and
report code. Third-party source tables, ORCS/DepMap arrays and feature arrays are not redistributed.
Their availability and permissions remain those of the original providers. The preparation guide
records source-member requirements, the PDF extraction limitation and the mutable upstream source
locator. Author blocks reproduce the supplied study donor sets; regeneration of author metadata is
a separate interface. No private names or overlap inventory are needed for these commands.

Focused adapter tests use explicitly manufactured outcomes and do not establish policy performance:

```bash
python -B -m pytest evaluation/test_evaluation.py -q
```
