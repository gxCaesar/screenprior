"""Reading the corpus, and refusing to guess.

A screen file that cannot be parsed stops the run here. That is a deliberate reversal of how the
exploratory scripts behaved, and the reason is specific to this kind of pipeline: the natural
shortcuts do not drop rows, they RELABEL them. `if row["HIT"] == "YES"` turns a missing or
malformed hit call into a negative, silently and always in the same direction, and no downstream
check would ever see it.

That hazard was measured before this module was written. Across 1,225 ORCS screens and 22,456,462
rows the HIT column contains exactly two values -- 'NO' (20,831,100) and 'YES' (1,625,362) -- and
zero rows take any of the silent paths: no undecodable bytes, no ragged rows, no missing symbols,
no unparseable scores. So the published numbers are untouched by it. `strict=True` is here to keep
it that way on the next release of the source data, which is where a change of this kind would
actually arrive.

BLOCKING IS PART OF READING THE DATA, not a downstream option, so it lives here. Two publications
contribute 325 and 340 screens to this corpus; a screen predicting its own siblings is a
within-paper split wearing the word transfer, and it is the easiest way to get a large and
completely fictitious result.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

__all__ = ["GENEDISCO_SIBLING_PMIDS", "Corpus", "ScreenParseError", "parent_cell_line",
           "read_screen"]

# GeneDisco assays that are themselves in ORCS. Dropped from the corpus rather than merely blocked,
# so that anything built from it stays usable as a GeneDisco training set.
GENEDISCO_SIBLING_PMIDS = {
    "35113687",   # Schmidt et al., ORCS screens 2421-2427 -- the IFNg and IL2 assays
    "31921143",   # Zhuang et al., ORCS screens 1081-1083 -- the NK assay
    "33574281",   # Zhu et al., ORCS screen 1622 -- the SARS-CoV-2 assay
}

REQUIRED = ("OFFICIAL_SYMBOL", "HIT", "SCORE.1")


def parent_cell_line(name: str) -> str:
    """The parent line a recorded ORCS cell-line string belongs to.

    ORCS records sublines and spellings as distinct strings, so blocking on the strings keeps
    HEK293T among HEK-293's donors. Case and punctuation are dropped, every T-cell description
    becomes TCELL, and otherwise the leading letters and first run of digits are kept: HEK293T and
    HEK-293 become HEK293, A549-ACE2 becomes A549, A2780cis becomes A2780. Measured on the
    1,217-screen corpus: 689 names become 670 parents, and the donor sets of 17 of the 153 targets
    change.
    """
    u = re.sub(r"[^A-Z0-9]", "", str(name).upper())
    if "TCELL" in u:
        return "TCELL"
    m = re.match(r"^([A-Z]+\d+)", u)
    return m.group(1) if m else u
HIT_VALUES = ("YES", "NO")


class ScreenParseError(ValueError):
    """Raised instead of dropping, converting or relabelling an input row."""


def read_screen(path, strict: bool = True):
    """Return (symbols, hits, scores) for one ORCS screen file.

    strict=True refuses anything it cannot read exactly: a byte that is not UTF-8, a row whose
    field count differs from the header's, an empty gene symbol, a HIT value outside {YES, NO}, or
    a SCORE.1 that is neither blank, '-' nor a float. strict=False exists for triaging a new data
    release and must not be used to produce a reported number.
    """
    path = Path(path)
    raw = path.read_bytes()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        if strict:
            raise ScreenParseError(f"{path.name}: not valid UTF-8 ({exc})") from exc
        text = raw.decode("utf-8", errors="replace")

    lines = text.splitlines()
    if not lines:
        raise ScreenParseError(f"{path.name}: empty file")
    header = lines[0].lstrip("#").split("\t")
    absent = [name for name in REQUIRED if name not in header]
    if absent:
        raise ScreenParseError(f"{path.name}: missing column(s) {absent}")
    width = len(header)
    where = {name: header.index(name) for name in REQUIRED}

    symbols, hits, scores = [], [], []
    for number, line in enumerate(lines[1:], start=2):
        if not line:
            continue
        parts = line.split("\t")
        if len(parts) != width:
            if strict:
                raise ScreenParseError(
                    f"{path.name}:{number}: {len(parts)} fields, header has {width}")
            continue
        symbol = parts[where["OFFICIAL_SYMBOL"]].strip().upper()
        if not symbol:
            if strict:
                raise ScreenParseError(f"{path.name}:{number}: empty OFFICIAL_SYMBOL")
            continue
        call = parts[where["HIT"]].strip()
        if call not in HIT_VALUES:
            if strict:
                raise ScreenParseError(
                    f"{path.name}:{number}: HIT is {call!r}, expected one of {HIT_VALUES}. "
                    "Reading this as a non-hit would relabel it, so the run stops instead.")
            call = "NO"
        written = parts[where["SCORE.1"]].strip()
        if written in ("", "-"):
            value = np.nan
        else:
            try:
                value = float(written)
            except ValueError as exc:
                if strict:
                    raise ScreenParseError(
                        f"{path.name}:{number}: SCORE.1 is {written!r}") from exc
                value = np.nan
        symbols.append(symbol)
        hits.append(call == "YES")
        scores.append(value)
    return symbols, np.array(hits, dtype=bool), np.array(scores, dtype=np.float64)


@dataclass
class Corpus:
    """A gene x screen hit matrix with its tested mask and the metadata blocking needs.

    `tested` is not redundant with `hit`. A gene absent from a screen's library is not a measured
    non-hit, and collapsing the two makes library composition look like biology -- which matters
    here: of the 1,225 screens that pass the filters, 1,182 use a nuclease and 27 an
    activator, across 691 cell lines.
    """

    genes: np.ndarray
    screen_id: np.ndarray
    pmid: np.ndarray
    cell_line: np.ndarray
    phenotype: np.ndarray
    hit: np.ndarray          # (n_genes, n_screens) bool
    tested: np.ndarray       # (n_genes, n_screens) bool

    def __post_init__(self) -> None:
        n_genes, n_screens = self.hit.shape
        if self.tested.shape != self.hit.shape:
            raise ValueError("hit and tested matrices differ in shape")
        if len(self.genes) != n_genes:
            raise ValueError(f"{len(self.genes)} gene labels for {n_genes} rows")
        for name in ("screen_id", "pmid", "cell_line", "phenotype"):
            if len(getattr(self, name)) != n_screens:
                raise ValueError(
                    f"{name} has {len(getattr(self, name))} entries for {n_screens} screens")
        if (self.hit & ~self.tested).any():
            raise ValueError("a hit is recorded for a gene the screen did not test")

    @property
    def promiscuity(self) -> np.ndarray:
        """Fraction of screens that called each gene a hit, among those that tested it.

        The label-free column that beats DepMap essentiality on 136 of 153 held-out screens. It is
        not a popularity index: the number of screens that tested a gene scores at chance on its
        own (AUPRC lift 1.044), and the two are essentially uncorrelated (Spearman -0.063).
        """
        return self.hit.sum(axis=1) / np.maximum(self.tested.sum(axis=1), 1)

    def donors_for(self, target: int, block_cell_line: bool = False) -> np.ndarray:
        """Every screen usable as a donor for `target`. Blocking by publication is mandatory.

        `block_cell_line` additionally removes screens of the same cell line. That is the sharper
        control -- 33 screens here are K-562 and 43 are hTERT-RPE1 -- and it moved the headline
        margin from +0.0205 to +0.0199 on the strings as recorded, and to +0.0198 after
        `with_parent_cell_lines`, i.e. not at all.
        """
        usable = self.pmid != self.pmid[target]
        if block_cell_line:
            usable &= self.cell_line != self.cell_line[target]
        if not usable.any():
            raise ValueError(f"no donors remain for screen {self.screen_id[target]}")
        return np.flatnonzero(usable)

    def with_parent_cell_lines(self) -> Corpus:
        """The same corpus with each cell line replaced by its `parent_cell_line`."""
        parents = np.array([parent_cell_line(c) for c in self.cell_line])
        return Corpus(self.genes, self.screen_id, self.pmid, parents, self.phenotype, self.hit,
                      self.tested)

    def drop_pmids(self, pmids) -> Corpus:
        """Remove whole publications.

        Used for the GeneDisco siblings, which must not merely be blocked: a corpus that still
        contains them cannot be used to train anything evaluated on GeneDisco.
        """
        keep = ~np.isin(self.pmid, list(pmids))
        return Corpus(self.genes, self.screen_id[keep], self.pmid[keep], self.cell_line[keep],
                      self.phenotype[keep], self.hit[:, keep], self.tested[:, keep])

    def targets_one_per_publication(self) -> np.ndarray:
        """One target screen per publication, chosen deterministically.

        The publication is the honest unit, not the screen: reporting 1,225 screens from 167 papers
        as 1,225 independent evaluations would overstate n more than sevenfold. Ties break by the
        largest tested universe, then the lowest screen id, so the choice does not depend on the
        order files happened to be read in.
        """
        order = np.lexsort((np.array([int(s) for s in self.screen_id]), -self.tested.sum(axis=0)))
        seen, targets = set(), []
        for column in order:
            if self.pmid[column] not in seen:
                seen.add(self.pmid[column])
                targets.append(int(column))
        return np.array(targets)

    def save(self, path) -> None:
        np.savez_compressed(
            path, genes=self.genes, screen_id=self.screen_id, pmid=self.pmid,
            cell_line=self.cell_line, phenotype=self.phenotype,
            hit=self.hit.astype(np.uint8), tested=self.tested.astype(np.uint8))

    @classmethod
    def load(cls, path) -> Corpus:
        cache = np.load(path, allow_pickle=False)
        return cls(cache["genes"], cache["screen_id"], cache["pmid"], cache["cell_line"],
                   cache["phenotype"], cache["hit"].astype(bool), cache["tested"].astype(bool))
