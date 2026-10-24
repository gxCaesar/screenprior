# ScreenPrior

ScreenPrior is a Python package for batch active learning in genome-scale CRISPR screens. It uses
completed screens as a prior over the running screen. Donor weights depend on the hit labels
acquired so far. A bagged ridge surrogate ranks genes offered by the sampled donors.

The package contains the core policies, data readers, gene representations, metrics, command
line interface, an external-target adapter, preparation scripts for the selected 40 external
screens and saved-result reporters. The companion study-data archive supplies the scientific
result rows, diagnostics and plotted quantities. It does not include original third-party
gene tables, tested/hit masks, feature arrays or checkpoints. Running a generated example checks
the software interface; reporting the saved study rows reproduces the stored analyses.

## Installation

Python 3.9 or later is required. The runtime dependencies are NumPy, pandas, SciPy, scikit-learn
and Typer. The core implementation runs on a CPU.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-lock.txt -r preparation/requirements.txt
python -m pip install -e .
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
COLUMNS=240 NO_COLOR=1 python -m pytest -q \
  tests preparation/test_preparation.py evaluation/test_evaluation.py
```

`requirements-lock.txt` supplies the tested CPU dependency pins. Package metadata in
`pyproject.toml` declares the supported ranges. Click is pinned separately for Python below 3.10
and Python 3.10 or later to meet Typer's requirements. The PDF preparation reader additionally
needs Poppler's `pdftotext` executable; see [preparation/USAGE.md](preparation/USAGE.md).

A fresh isolated wheel installation was checked on macOS arm64 with Python 3.10.20, NumPy 2.0.2,
pandas 2.3.3, SciPy 1.13.1 and scikit-learn 1.6.1. The combined core, adapter, preparation and
evaluation selection passed 181 tests. Ten additional saved-comparison tests passed after the
companion data were supplied, for 191 distinct tests. Run those with
`python -m pytest comparison/test_comparison.py -q` after placing the study data as below.

The corrected dependency installation took 15.43 s using a task-local cache populated during an
earlier failed dependency resolution. Total active environment and installation commands took
38.12 s including that correction. These are observations from one installation, not a cold
download time or a general hardware benchmark. The two generated examples below completed in
6.33 s and 11.01 s in that environment.

On that Mac, the pinned NumPy wheel used Accelerate and emitted floating-point warnings for
matrix products whose finite results agreed with known values. The tests and examples retained
their finite/value checks and passed with warnings visible. This diagnosis applies to the tested
cases; it is not assurance for arbitrary inputs or a reason to ignore nonfinite results.

## Reproduce the saved study reports

Extract `screenprior-code-20261024.zip` and `screenprior-study-data-20261024.zip` into the same
parent directory. Both use `screenprior/` as their top directory; the second places its contents
under `screenprior/data/study/`. From a repository clone, place that same `study/` directory under
`data/`. Run the following commands from the code root, using new output filenames:

```bash
python -B evaluation/adjudicate_a1.py \
  --rows data/study/external40/primary_rows.csv.gz --out primary-report.json

python -B evaluation/report_secondary.py \
  --core data/study/external40/all_core_rows.csv.gz \
  --free data/study/external40/all_free_controls.csv.gz \
  --status data/study/external40/condition_status.csv \
  --diagnostics-root data/study/external40/diagnostics --out secondary-report.json

python -B comparison/report_assayformer.py --out comparison-report.json
```

The external reports validate 36,000 primary rows, 105,300 total core rows, 2,574 label-free
control rows and 400 primary donor diagnostics for the selected 40 publications. There are
120 target-condition records: 117 completed conditions and three ineligible non-essential pools.
The reporters average seeds within each publication before inference. Their saved-result outputs
matched all 882 primary numeric fields and all 31,327 secondary numeric fields exactly. Missing
seeds, changed denominators and inconsistent diagnostics are refused.

The comparison reporter reads 200 ScreenPrior rows and 400 AssayFormer endpoint rows over
20 screens from 15 publication groups. It preserves the original screen-mean estimand and
publication-cluster interval, including the inconclusive comparison outcome. Its 29 original
numeric fields matched within absolute tolerance 1e-12. The nested never-read subset is not an
independent confirmation. See [comparison/USAGE.md](comparison/USAGE.md) for the full-library and
effective-budget denominators.

These commands recalculate reports from saved scientific results. They do not run acquisition
models, verify every original gene-level call or repeat checkpoint inference. The
[study-data guide](data/study/README.md), [field dictionary](data/study/DATA_DICTIONARY.md) and
[file catalogue](data/study/data-files.csv) describe the supplied results and figure-panel data.

## Synthetic example

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python examples/synthetic_pipeline.py --out demo-output
```

The example writes six ORCS-format screens from five synthetic publications and a synthetic
DepMap-format matrix. It invokes `build-corpus`, `build-static`, `run` and `report` through the
installed command line module. The run uses all nine core comparison arms, two seeds and three
batches of eight genes.

The output directory contains the generated inputs, aligned arrays, `results.csv`, `report.txt`
and `summary.json`. The summary should report five targets, 350 represented genes, 90 result rows
and 24 acquisitions per row. Numerical policy margins can depend on numerical library versions.
Use a new output directory when repeating the example.

## One external target

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
python examples/external_target.py --out external-demo
```

This example constructs four synthetic donor screens and appends exactly one external target.
An explicit publication block removes one donor, leaving three donor screens from two
publications. Both donors and target have separate tested and hit masks. The example keeps the
adapter's nine policy positions, ten seeds, ten batches of 64 and common bagged-ridge fitting
recipe. It supplies a rank-2 donor embedder through the supported callable argument because its
generated donor corpus is small. The adapter's default donor embedding rank is 64.

The generated target has 5,148 tested genes among 5,200 represented genes. Expected software
output is 900 cycle rows, 90 complete acquisition traces, 22 label-free control rows and ten
post-cycle-3 donor diagnostics. The output directory contains input arrays, `sequential.csv`,
`controls.csv`, `diagnostics.csv`, ordered trace archives, donor weights and `summary.json`.
These counts check the interface. The synthetic policy scores are not experimental evidence.

For new inputs, use the Python API in `screenprior.external`:

```python
from screenprior.external import external_corpus, label_free_baselines, run_one_target

one = external_corpus(
    donors, target_tested, target_hit, pub_id="external-publication",
    screen_id="external-screen", metadata={"cell_line": "cell-line", "phenotype": "response"},
    blocked_donor_publications=blocked_publications,
)
target = len(one.screen_id) - 1
sequential = run_one_target(one, target, features, gene_effect, pub_id="external-publication")
controls = label_free_baselines(one, target, gene_effect, pub_id="external-publication")
```

`donors` is an aligned `Corpus`. The binary `target_tested` and `target_hit` vectors have one entry
per corpus gene, in that exact order. Hits on untested genes are refused. `features` is a finite
gene-by-feature static matrix, and `gene_effect` is a finite vector in the same gene order.
Metadata may include a PMID or DOI, cell line and phenotype. Map source identifiers and apply the
original screen authors' hit rule before calling the adapter. It does not infer hit thresholds
or map gene aliases.

Supply a set of donor publication identifiers to `blocked_donor_publications`, including an
explicit empty set when no additional publications should be removed. The caller determines
author overlap or other additional exclusions. The adapter removes every donor screen from
each listed publication, retains the package's GeneDisco exclusions and refuses unknown blocks.
It rejects a target already present in the donor corpus and prevents an appended external
target from becoming a donor for another external target. Start each target from the same
unaltered donor corpus. The donor embedding is built after exclusions and publication blocking.

`run_one_target` uses the full pool by default. Its fixed design is ten seeds, 64 acquisitions
per cycle, ten cycles and the original nine policy slots. The core eligibility thresholds are
5,000 candidate genes and 30 hits. Ineligible targets return `status="not_eligible"` and empty
outputs. Malformed inputs, empty pools and pools with no hits raise an error. The supported
`embed_donors` callable and `model` arguments allow controlled interface tests without changing
the default design.

Returned `rows` contain each cycle's N, H, donor count, acquired-hit count, recall and chance.
Returned `traces` retain acquisition order and acquired labels. `diagnostics` report donor
agreement, weights and effective donor count after the first 192 ScreenPrior acquisitions,
using inverse temperature 1.0. This post-acquisition diagnostic describes the updated donor
weights after the third batch. It consumes no random draws.

The separate `label_free_baselines` function ranks by donor hit frequency and negative DepMap
gene effect. Frequency divides donor hits by the number of donors that tested each gene.
Each score is evaluated as a pure ranking and after the same shared opening used by the core.
The defaults produce two pure rows (`seed=-1`, `cycle=0`) and twenty shared-opening rows.
These controls do not fit a surrogate or use target labels to select genes. Their `budget`,
`batch` and `seeds` arguments also support small known-answer examples.

## Data access and preparation

The real pipeline takes the human BioGRID ORCS 2.0.18 archive and DepMap 21Q4
`CRISPR_gene_effect.csv`. Download these from the providers listed in
[data/manifests/sources.yaml](data/manifests/sources.yaml). Source data are not redistributed.
[data/README.md](data/README.md) describes input selection, preparation and additional source
references.

Keep the human archive's screen files and its `SCREEN_INDEX` together. Human and mouse archives
can use the same index filename. The parser checks the index's `ORGANISM_OFFICIAL` column.
Preserve the DepMap provider's filename because the reader uses it to identify matrix orientation.

```bash
screenprior build-corpus --orcs inputs/orcs-human --index inputs/orcs-human/SCREEN_INDEX.index.tab.txt \
    --out corpus.npz
screenprior build-static --depmap inputs/CRISPR_gene_effect.csv --corpus corpus.npz \
    --out-corpus corpus.depmap.npz --features features.npy --gene-effect gene-effect.npy
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
screenprior run --corpus corpus.depmap.npz --features features.npy --gene-effect gene-effect.npy \
    --policy random --policy greedy --policy topuncertain --policy softuncertain \
    --policy coreset --policy badge --policy task-thompson \
    --policy task-thompson/permuted --policy task-thompson/uniform \
    --batch 64 --cycles 10 --seeds 3 --pool full --out results.csv
screenprior report --results results.csv --against badge --n-candidates 1
```

Replace the index path with the filename actually supplied in the human archive. The last
command's `--n-candidates` is the number of candidates searched in the analysis being reported.
The value one above represents a run without candidate selection. It must be changed when
selection occurred. `report` is a general paired summary, rather than a confirmation-specific
statistical analysis.

`build-corpus` writes a gene-by-screen hit matrix and a separate tested mask. Its default filters
require at least 12,000 tested genes, at least 30 hits and a hit rate no greater than 0.20. It
removes the GeneDisco sibling publications listed in `screenprior.corpus`. `build-static` writes
three aligned outputs after restricting the corpus to genes represented by DepMap. Pass its
`--out-corpus` output to `run`, rather than the unrestricted input corpus.

`run` selects one target per publication, using the largest tested universe and then the lowest
numeric screen identifier. It blocks donor screens from that publication and rebuilds the corpus
embedding after blocking. The default pool is `non-essential`, which removes genes with mean
DepMap effect below −0.5. The command above explicitly chooses `full`. Additional author-group
exclusions for an external target are supplied explicitly through `screenprior.external`, as
described above. The command line interface does not fetch authors or construct that exclusion set.

For the selected 40 external publications, [preparation/USAGE.md](preparation/USAGE.md)
provides the original source locators, literal hit rules, gene mapping and reconstruction
command. It reads user-supplied tables and produces separate tested, hit and represented-pool
masks in the documented gene order. The accompanying publication-ID block lists specify the
author-group exclusions used for those targets. Raw tables and source-derived masks are not
included. The preparation dependencies are listed separately in
[preparation/requirements.txt](preparation/requirements.txt).

The source-to-acquisition interface is supplied by [evaluation/USAGE.md](evaluation/USAGE.md).
After obtaining the licensed originals and generating the documented full and represented
corpora, prepare the masks and run the external study with:

```bash
python -B preparation/prepare_targets.py --source-root /path/to/licensed-sources \
  --corpus /path/to/corpus.npz --represented-corpus /path/to/corpus.depmap.npz \
  --out /path/to/new-targets.npz

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
python -B evaluation/run_study.py --corpus /path/to/corpus.depmap.npz \
  --features /path/to/features.npy --gene-effect /path/to/gene-effect.npy \
  --targets-npz /path/to/new-targets.npz --out /path/to/new-study-output
```

An optional local source-map file resolves filenames and nested members as described in the
preparation guide. The runner checks the 40 target identities, exact represented-gene projection,
source counts and fixed publication-ID exclusion lists before dispatching fits. It runs the
full author-blocked primary, non-essential sensitivity and author-blocking-off sensitivity with
the fixed nine arms, ten seeds and ten batches of 64, plus four label-free control labels.
Use `--check-inputs` instead of `--out` to validate supplied arrays without fitting. This interface
was verified against the actual prepared inputs; a fresh 40-publication model replay is not part
of the release validation.

## Policies and metrics

The ScreenPrior policy is exposed under the stable API name `task-thompson`. Report its removal
ablation, `task-thompson/uniform`, and matched control, `task-thompson/permuted`, alongside it.
All three use the same acquisition code, surrogate specification and fixed inverse temperature
of 1.0. Each arm fits its own acquired labels once the acquisition paths diverge.

The registry also provides the classical acquisition policies and the equation-based DiscoBAX
implementation described in `NOTICE`. The name `badge` denotes the farthest-first variant.
`badge/kmeans++` implements k-means++ selection from the gradient embedding. These are distinct
arms and should be named explicitly in a comparison.

Recall divides the number of acquired hits by all hits in the candidate pool. Precision divides
it by the acquisition budget. AUPRC lift divides average precision by the pool's hit prevalence.
Metrics on different candidate pools are not interchangeable.

## Reproducibility and limitations

Set the BLAS thread variables before importing NumPy. The order of policies is part of the
experiment because each policy uses `seed * 31 + 101 * slot`. The nine-arm order in the command
above fixes these slots. Append additional arms rather than reordering existing ones.

The historical harness initializes the shared opening with `default_rng(seed)` and each policy
with `seed * 31 + 101 * slot`. For canonical `random` at seed zero, both initializations reuse
seed zero, so the continuation restarts the opening stream. This behavior is preserved in this
core implementation.

`screenprior.external.run_one_target` instead substitutes only the random arm with
`screenprior.random_control.corrected_random_policy`. Its generator consumes the original
opening and then continues from that state. The eight other arms retain their original slot
streams. The lower-level `continuous_random_indices` helper exposes the same continuous
sampling path. A fresh random-policy closure is required for each target and seed. Targets of
equal pool size still share a seed's positional path. An observed recall-minus-chance interval
is descriptive and is not an algorithmic proof of uniform sampling.

The surrogate is fitted on acquired labels at every cycle. The package does not require an
offline trained screen model. Corpus hit calls and gene identifiers must still be checked for
the intended application. A gene absent from a donor's library remains untested rather than
being relabelled as a non-hit.

The source package and companion study data support the saved 40-publication reports and the
matched-pool saved-row comparison. The preparation and evaluation directories additionally
provide the route from user-supplied licensed sources to a new external acquisition run. Fixed
publication-ID block lists are supplied; author metadata are not fetched or reconstructed.

The earlier 153-publication development campaign is not fully rerunnable from this release.
Its included panel exports support inspection and redrawing of the stored quantities. The
saved comparison cannot independently certify gene-level pool membership or reproduce model
checkpoint inference without the original libraries and model resources. Provider access,
source-member gaps and the documented PDF-extraction limitation remain relevant to source
reconstruction. See [data/README.md](data/README.md) and the component usage documents.

The maintained source repository is [gxCaesar/screenprior](https://github.com/gxCaesar/screenprior). The companion generated-data archive and code-archive identifiers will be added after those records are published.

## Licence and citation

The code is distributed under the MIT licence in [LICENSE](LICENSE). Third-party method
attribution and the DiscoBAX implementation scope are described in [NOTICE](NOTICE). Data retain
their providers' terms. The separate study-data licence is recorded in its own guide and is not
conferred by the code licence. Software citation metadata are in [CITATION.cff](CITATION.cff)
and [CITATION.md](CITATION.md). Public repository and archive identifiers are added when those
records exist.
