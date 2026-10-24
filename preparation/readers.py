"""Read the selected author tables without changing their call rules.

User-supplied files are resolved by publication and asset name. The selected table, sheet
and literal call rule are fixed in targets.csv; incomplete inputs are refused.
"""

import gzip
import re
import subprocess
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from source_files import Tables


@dataclass
class RawScreen:
    rows: pd.DataFrame
    sources: list
    count_note: str = ""


def numeric(frame, name):
    return pd.to_numeric(frame[name], errors="coerce")


def raw_rows(ids, scored, hits):
    """Keep source rows until the assembly stage performs mapped-gene OR collapsing."""
    frame = pd.DataFrame({"raw_id": list(ids), "scored": list(scored), "hit": list(hits)})
    frame["scored"] = frame.scored.fillna(False).astype(bool)
    frame["hit"] = frame.hit.fillna(False).astype(bool)
    if (frame.hit & ~frame.scored).any():
        raise ValueError("source hit lacks a measured score")
    return frame


def _xml_sheet(table, name, strings):
    """Preserve sparse Excel cell positions instead of shifting cells across model columns."""
    ns = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
    output = []
    for _, row in ET.iterparse(table.path(name)):
        if row.tag != ns + "row":
            continue
        values = {}
        for cell in row.findall(ns + "c"):
            column = re.match(r"[A-Z]+", cell.attrib["r"])[0]
            index = 0
            for char in column:
                index = 26 * index + ord(char) - ord("A") + 1
            value = cell.find(ns + "v")
            if value is not None:
                values[index - 1] = (strings[int(value.text)] if cell.get("t") == "s"
                                     else value.text)
        record = [values.get(i) for i in range(max(values, default=-1) + 1)]
        output.append(record)
        row.clear()
    return output


def _um_screen(table):
    ns = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
    prefix = "41588_2025_2222_MOESM4_ESM.xlsx#member="
    strings = []
    with gzip.open(table.path(prefix + "xl_sharedStrings.xml.gz"), "rb") as fh:
        for _, element in ET.iterparse(fh):
            if element.tag == ns + "si":
                strings.append("".join(t.text or "" for t in element.iter(ns + "t")))
                element.clear()
    all_rows = _xml_sheet(table, prefix + "xl_worksheets_sheet10.xml", strings)
    headers = all_rows[1]
    model_col = headers.index("921")
    gene_rows = [(r[0], r[model_col] if len(r) > model_col else None)
                 for r in all_rows[2:] if r and r[0]]
    hit_rows = _xml_sheet(table, prefix + "xl_worksheets_sheet8.xml", strings)
    hits = {r[1] for r in hit_rows[5:] if len(r) >= 5 and r[2] == "921"
            and str(r[4]) == "1"}
    ids = [r[0] for r in gene_rows]
    # The author hit column is authoritative, including its explicitly documented rounding rows.
    return raw_rows(ids, [v not in (None, "") for _, v in gene_rows],
                    [g in hits for g in ids])


def _pdf_table(table, output_dir):
    """Extract the supplied PDF with the recorded layout-token parser."""
    pdf = table.path("cancers-3260552-supplementary.pdf")
    if not pdf.is_file():
        raise FileNotFoundError(pdf)
    table.sources.append(str(pdf))
    text = Path(output_dir) / "pdf-extraction" / "cancers-3260552-supplementary.txt"
    text.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["pdftotext", "-layout", str(pdf), str(text)], check=True)
    lines = text.read_text(encoding="utf-8", errors="replace").splitlines()
    i = next(k for k, line in enumerate(lines) if "Transcript_name" in line)
    j = next(k for k, line in enumerate(lines) if k > i and "Supplementary Table S4." in line)
    tokens = [t for line in lines[i:j] for t in re.split(r"\s{2,}", line.strip()) if t]
    name_re = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.\-]*$")
    number_re = re.compile(r"^-?\d+(?:,\d+)?$")
    records, k = [], 0
    while k < len(tokens) - 7:
        values = tokens[k + 1:k + 8]
        if (name_re.match(tokens[k]) and "__" in tokens[k]
                and all(number_re.match(value) for value in values)):
            records.append((tokens[k].split("__")[0],
                            *[float(value.replace(",", ".")) for value in values[1:]]))
            k += 8
        else:
            k += 1
    return raw_rows([r[0] for r in records], [True] * len(records),
                    [r[6] < 0.05 and r[5] < 0 for r in records])


def _crc9_screen(table, selected):
    """Stream the large author's OOXML sheet while retaining only gene/p/FDR for this target."""
    path = table.path("mmc3.xlsx")
    key = selected["screen_label"].split(" in ")[0]
    ns = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
    strings, rows, header = [], [], None
    with zipfile.ZipFile(path) as archive:
        with archive.open("xl/sharedStrings.xml") as fh:
            for _, item in ET.iterparse(fh):
                if item.tag == ns + "si":
                    strings.append("".join(part.text or "" for part in item.iter(ns + "t")))
                    item.clear()
        with archive.open("xl/worksheets/sheet1.xml") as fh:
            for _, item in ET.iterparse(fh):
                if item.tag != ns + "row":
                    continue
                cells = {}
                for cell in item.findall(ns + "c"):
                    value = cell.find(ns + "v")
                    if value is not None:
                        col = re.match(r"[A-Z]+", cell.attrib["r"])[0]
                        cells[col] = (strings[int(value.text)] if cell.attrib.get("t") == "s"
                                      else value.text)
                if header is None:
                    header = {name: col for col, name in cells.items()}
                    gene_col = header["gene"]
                    p_col = header.get(key + "_pvalue", header.get(key + "_Pval"))
                    fdr_col = header[key + "_fdr"]
                    if p_col is None:
                        raise ValueError(f"selected comparison {key} has no pvalue/Pval column")
                else:
                    rows.append((cells.get(gene_col), cells.get(p_col), cells.get(fdr_col)))
                item.clear()
    f = pd.DataFrame(rows, columns=["gene", "p", "fdr"])
    p, fdr = numeric(f, "p"), numeric(f, "fdr")
    cut = .20 if key.startswith("CRC_9") else .10
    return raw_rows(f.gene, p.notna(), (p < .05) & (fdr < cut))


def read_screen(pub_id, selected, assets, output_dir):
    """Read exactly the table specified for a selected publication."""
    t = Tables(assets, pub_id)
    note = ""
    if pub_id == "pmid_40615675":
        rows = _um_screen(t)

    elif pub_id == "pmid_39518122":
        rows = _pdf_table(t, output_dir)

        note = "PDF text loss documented in batch3; transcript-level 588 vs printed 590"

    elif pub_id == "pmid_41882347":
        f = t.excel("pmid_41882347_MOESM3.xlsx", "P value")

        col = "Post-stim Surface FPR1 Screen"

        if col not in f:
            col = next(c for c in f if "Post-stim" in c)

        hits = set()

        for n in (6, 7):
            g = t.excel(f"pmid_41882347_MOESM{n}.xlsx", "Sheet1")
            hits |= set(g.Gene)

        rows = raw_rows(f["Gene Name"], numeric(f, col).notna(), f["Gene Name"].isin(hits))

    elif pub_id == "pmid_42123509":
        f = t.excel("Table_S1_genome-wide_CRISPR_screen_data.xlsx", "Gene_score",
                    need=["Gene", "tRRA score", "p-value"])

        score = numeric(f, "tRRA score")

        rows = raw_rows(f.Gene, score.notna(), (score.abs() >= 1) & (numeric(f, "p-value") < .05))

    elif pub_id == "pmid_40493189":
        f = t.csv("pnas.2424985122.sd03.csv")

        cols = [c for c in f if c.startswith("HCC44") and c.lower().endswith(".fdr")]

        scores = f[cols].apply(pd.to_numeric, errors="coerce")

        rows = raw_rows(f.Gene, scores.notna().any(axis=1), (scores < .1).any(axis=1))

    elif pub_id == "pmid_41579376":
        f = t.excel("NIHMS2151702-supplement-2.xlsx", "vishal_crispr_1020.mageck_mle.g",
                    need=["Gene", "LH2_top10|beta", "LH2_top10|wald-p-value"])

        rows = raw_rows(f.Gene, numeric(f, "LH2_top10|beta").notna(),
                        numeric(f, "LH2_top10|wald-p-value") < .05)

    elif pub_id == "pmid_39117798":
        f = t.excel("41375_2024_2325_MOESM4_ESM.xlsx", "T14DMSO_VS_T0", header=1)

        rows = raw_rows(f.id, numeric(f, "neg|p-value").notna(), numeric(f, "neg|fdr") < .1)

    elif pub_id == "pmid_42793135":
        f = t.excel("PMC13604100_supplementaryFiles.zip", "Sheet1", need=["id", "neg|score"],
                    members=("biomolecules-16-01303-s001.zip", "Supplementary Table S1.xlsx"))

        rows = raw_rows(f.id, numeric(f, "neg|score").notna(), numeric(f, "neg|score") < .01)

    elif pub_id == "pmid_38229201":
        g = t.csv("GSE232614_WSU.gene_summary.txt.gz", sep="\t")

        s = t.csv("GSE232614_WSU.sgrna_summary.txt.gz", sep="\t")

        score, abundance = numeric(s, "score"), numeric(s, "treat_mean")

        hits = (set(s.loc[score >= score.quantile(.99), "Gene"])
                & set(s.loc[abundance.rank(ascending=False, method="first") <= 3000, "Gene"]))

        rows = raw_rows(g.id, numeric(g, "pos|p-value").notna(),
                        g.id.isin(hits) & (numeric(g, "pos|p-value") <= .005))

    elif pub_id == "pmid_39894896":
        f = t.excel("PMC11788431/supplementaryFiles.zip",
                    members=("41467_2025_56632_MOESM3_ESM.xlsx",))

        h = t.excel("PMC11788431/supplementaryFiles.zip",
                    members=("41467_2025_56632_MOESM4_ESM.xlsx",))

        counts = h.groupby("Symbol").size()

        hits = set(counts[counts >= 4].index)

        rows = raw_rows(f.Symbol, numeric(f, "count.sortedB.vs.count.unsorted.p.value").notna(),
                        f.Symbol.isin(hits))

    elif pub_id == "pmid_41397982":
        f = t.excel("41467_2025_67236_MOESM4_ESM.xlsx", selected["sheet"])

        rows = raw_rows(f.id, numeric(f, "neg|score").notna(), numeric(f, "pos|score") < .01)

    elif pub_id == "pmid_42212547":
        f = t.excel("MOL2-9999-0-s004.xlsx", "DrugZ")

        z = numeric(f, "normZ")

        rows = raw_rows(f.GENE, z.notna(), (numeric(f, "fdr_synth") < .1) & (z < 0))

    elif pub_id == "pmid_38315730":
        f = t.excel("S1_Table_more_aggregation.xlsx", "res_df_more_agg")

        rows = raw_rows(f.gene_name, numeric(f, "mean_val").notna(),
                        (numeric(f, "mean_val") > 1) & (numeric(f, "mean_pv") < .005))

    elif pub_id == "pmid_41886504":
        f = t.excel("pmid_41886504_supp.zip", "Fig.1", header=1,
                    members=("szag012_supplementary_data.zip", "Source_Data.xlsx"))

        value = numeric(f, "RRA score")

        n = f.loc[value.notna(), "id"].nunique()

        rows = raw_rows(f.id, value.notna(), value.rank(method="first") <= np.floor(.01 * n))

    elif pub_id == "pmid_39251587":
        f = t.excel("41467_2024_52176_MOESM10_ESM.xlsx", "Sheet1")

        score = numeric(f, "RPE1-SS48")

        rows = raw_rows(f["Gene name"], score.notna(), score < -1)

    elif pub_id == "pmid_41383979":
        frames = []

        for n, direction in ((1, "pos"), (2, "neg")):
            f = t.excel("pmid_41383979_supp.zip", f"Supplemental Table S{n}", header=1,
                        members=(f"Supplementary_Data{n}.xlsx",))
            p = numeric(f, f"{direction}|p-value")
            frames.append(raw_rows(f.id, p.notna(), p < .05))

        rows = pd.concat(frames, ignore_index=True)

    elif pub_id == "pmid_41756760":
        f = t.excel("PMC12932945_supp.zip", "Sheet1",
                    members=("vdag002_supplementary_data.zip", "Table S1.xlsx"))

        p = numeric(f, "pos|fdr")

        rows = raw_rows(f.Gene, p.notna(), p < .2)

    elif pub_id == "pmid_40021625":
        f = t.excel("41467_2025_57185_MOESM4_ESM.xlsx", "6-day")

        rows = raw_rows(f.id, numeric(f, "neg|score").notna(),
                        (numeric(f, "neg|rank") <= 191) | (numeric(f, "pos|rank") <= 191))

    elif pub_id == "pmid_39753722":
        f = t.excel("43018_2024_882_MOESM3_ESM.xlsx", "Suppl.Table4", header=4)

        blocks, hits = [], []

        for start in (1, 9):
            block = f.iloc[:, start:start + 6].copy()
            block.columns = ["gene", "score", "p", "sig", "n", "nsig"]
            hit = ((block.sig.astype(str).str.lower() == "true")
                   & (numeric(block, "nsig") >= 2))
            hits.append(set(block.loc[hit, "gene"]))
            blocks.append(block)

        hit = hits[0] & hits[1]

        rows = raw_rows(blocks[0].gene, numeric(blocks[0], "score").notna(),
                        blocks[0].gene.isin(hit))

    elif pub_id == "pmid_38480701":
        f = t.excel("41467_2024_46495_MOESM5_ESM.xlsx", "Suppl Table 1", header=2)

        beta, fdr = numeric(f, "treatment|beta"), numeric(f, "treatment|fdr")

        rows = raw_rows(f.Gene, beta.notna(), (fdr < .05) & (beta < -.5))

    elif pub_id == "pmid_38653965":
        frames, sets = [], {}

        for sheet in ("AP300", "LDK300", "AP750", "LDK750"):
            f = t.excel("41467_2024_47771_MOESM5_ESM.xlsx", sheet)
            fd = numeric(f, "FD")
            counts = f.loc[fd > 1.8].groupby("Gene").size()
            sets[sheet] = set(counts[counts >= 2].index)
            frames.append(raw_rows(f.Gene, fd.notna(), [False] * len(f)))

        hit = (sets["AP300"] | sets["AP750"]) & (sets["LDK300"] | sets["LDK750"])

        rows = pd.concat(frames, ignore_index=True)

        rows["hit"] = rows.raw_id.isin(hit) & rows.scored

    elif pub_id == "pmid_41887225":
        f = t.csv("jacquere_A375_mageck_RRA_results.gene_summary.txt", sep="\t")

        rows = raw_rows(f.id, numeric(f, "neg|score").notna(), numeric(f, "neg|fdr") < .05)

    elif pub_id == "pmid_40327396":
        f = t.excel("PMC12226049_SupplementaryFiles.zip", "Table_4",
                    members=("jciinsight-10-188051-s010.xlsx",))

        score = numeric(f, "DAY_14_DMSO|beta")

        rows = raw_rows(f.Gene, score.notna(), score.abs() >= 1)

    elif pub_id == "pmid_41564866":
        rows = _crc9_screen(t, selected)

    elif pub_id == "pmid_39198401":
        f = t.excel("PMC11358519_SupplementaryFiles.zip", "Supplementary Data 2", header=2,
                    members=("41467_2024_51646_MOESM5_ESM.xlsx",))

        score_cols = ["DMSO Beta Score", "AZA Beta Score"]

        rows = raw_rows(f.Gene, f[score_cols].notna().any(axis=1),
                        f.group.astype(str).str.strip().isin(["bottomcenter", "topcenter"]))

    elif pub_id == "pmid_42469227":
        f = t.excel("41392_2026_2847_MOESM2_ESM.xlsx", "Fig. 6B, 6D, S16")

        score = numeric(f, "control|beta")

        rows = raw_rows(f.Gene, score.notna(), score <= -.64)

    elif pub_id == "pmid_37971291":
        f = t.csv("PMC10783524_SupplementaryFiles.zip", sep="\t",
                  members=("gkad1054_supplemental_files.zip",
                           "Mellor_et_al Table S8_cddp_treated_summary.txt"))

        p = numeric(f, "neg|p-value")

        rows = raw_rows(f.id, p.notna(), p < .01)

    elif pub_id == "pmid_42694111":
        f = t.excel("44400_2026_125_MOESM3_ESM.xlsx", "All Genes and amalgam ctls")

        p = numeric(f, "fdr")

        rows = raw_rows(f.gene, p.notna(), p < .1)

    elif pub_id == "pmid_39856062":
        f = t.excel("pmid_39856062/MOESM4.xlsx", "b6(+)_b6(-)_v3.genescores",
                    need=["Gene", "dGS", "p3"])

        score, p = numeric(f, "dGS"), numeric(f, "p3")

        top = (score.rank(method="first", ascending=False) <= 300)

        bottom = (score.rank(method="first", ascending=True) <= 300)

        rows = raw_rows(f.Gene, p.notna(), (top | bottom) & (p < .05))

        note = "The author shortlist contains 456 calls after the rank and nominal-p conditions."

    elif pub_id == "pmid_39272807":
        f = t.excel("Supplemental_Table_S1_TKOv3.xlsx", "1.5 Replicate 1 PR", header=1)

        gene, bf, fdr = f.iloc[:, 5], pd.to_numeric(f.iloc[:, 6], errors="coerce"), pd.to_numeric(
            f.iloc[:, 9], errors="coerce")

        rows = raw_rows(gene, bf.notna(), (bf > 0) & (fdr <= .05))

    elif pub_id == "pmid_40993380":
        f = t.excel("PMC13221092/supplementaryFiles.zip", "table S1_Mageck_data_analysis",
                    members=("NIHMS2172163-supplement-Supplementary_Information_Table_1.xlsx",))

        score = numeric(f, "pos|score")

        rows = raw_rows(f.id, score.notna(), score < 1e-5)

    elif pub_id == "pmid_39715471":
        f = t.excel("pmid_39715471_epmc_bundle.zip", "Supplementary Table S1", header=1,
                    members=("BLOODA_ADV-2024-014157-mmc2.xlsx",))

        h = t.excel("pmid_39715471_epmc_bundle.zip", "Supplementary Table S2", header=1,
                    members=("BLOODA_ADV-2024-014157-mmc2.xlsx",))

        hits = set(h.loc[h["AraC Response"].isin(["Resistant", "Sensitive"]), "Gene"].astype(str)
                   .str.replace("__", "-", regex=False))

        rows = raw_rows(f.Gene, numeric(f, "AraC Diff-Beta").notna(), f.Gene.isin(hits))

    elif pub_id == "pmid_38939106":
        f = t.excel("pmid_38939106_epmc_bundle.zip", "Fig. 1B OE screen", members=("mmc2.xlsx",))

        score = [c for c in f if c.startswith("SigmaFC")]

        sig = [c for c in f if c.startswith("significant")]

        hit = f[sig].astype(str).apply(lambda col: col.str.lower() == "true").any(axis=1)

        rows = raw_rows(f.iloc[:, 0], f[score].notna().any(axis=1), hit)

        note = "NO-TARGET controls are removed before gene counts."

    elif pub_id == "pmid_41820368":
        f = t.excel("41467_2026_70517_MOESM3_ESM.xlsx", "786O d25 vs. 786O+VHL d29", header=2)

        rows = raw_rows(f.Gene, numeric(f, "BF").notna(), numeric(f, "FDR") < .05)

    elif pub_id == "pmid_40601736":
        f = t.excel("sciadv.adu9104_tables_s1_to_s3.zip", "Screen Results",
                    members=("adu9104_table_s1.xlsx",))

        lfc, logp = numeric(f, "Average LFC"), numeric(f, "Average -log(p-values)")

        rows = raw_rows(f["Gene Symbol"], logp.notna() & lfc.notna(),
                        (logp > 1.30103) & (lfc != 0))

    elif pub_id == "pmid_42685209":
        f = t.excel("PMC13537241_suppl.zip", "Table S2", header=4,
                    members=("sciadv.aec7585_tables_s1_to_s5.zip", "aec7585_table_s2.xlsx"))

        nf, pf = numeric(f, "neg|fdr"), numeric(f, "pos|fdr")

        rows = raw_rows(f.id, nf.notna() | pf.notna(),
                        ((numeric(f, "neg|lfc") < -2) & (nf < .1))
                        | ((numeric(f, "pos|lfc") > 2) & (pf < .1)))

    elif pub_id == "pmid_41860961":
        f = t.excel("PMC13012052_supp.zip", "Dataset S1", members=("pnas.2532814123.sd01.xlsx",))

        p = numeric(f, "neg.p.value")

        rows = raw_rows(f["gene id"], p.notna(), p < .05)

    elif pub_id == "pmid_41974002":
        f = t.excel("can-25-3654_table_s1-s12_suppst1-12.xlsx", "Supplementary Table S8", header=1)

        p = numeric(f, "neg|fdr")

        rows = raw_rows(f.id, p.notna(), p < .05)

    elif pub_id == "pmid_38316463":
        f = t.excel("LSA-2023-02353_TableS1.xlsx", "T19vsT5")

        low, high = numeric(f, "fdr_synth"), numeric(f, "fdr_supp")

        rows = raw_rows(f.GENE, numeric(f, "normZ").notna(), (low < .1) | (high < .1))

    elif pub_id == "pmid_41285767":
        f = t.excel("41467_2025_65277_MOESM3_ESM.xlsx", selected["sheet"], header=1)
        q2 = numeric(f, "Enrichment_Qvalue (2 Reps)")
        q3 = numeric(f, "Enrichment_Qvalue (3 Reps)")
        scored = q2.notna() | q3.notna()
        hits = scored & ((q2 < .35) | (q3 < .35))
        if not f["Gene"].equals(f.iloc[:, 7]):
            raise ValueError("SD2's two gene columns differ; replicate rows misalign")
        rows = raw_rows(f["Gene"], scored, hits)
        control = re.compile(r"non-?target|^ntctrl$|safe-?harbou?r|^control|^nt$|"
                             r"^(?:egfp|lacz|luciferase)$|^no-?target$", re.IGNORECASE)
        rows = rows[rows.raw_id.notna()].copy()
        rows.raw_id = rows.raw_id.astype(str).str.strip()
        rows = rows[rows.raw_id.ne("") & ~rows.raw_id.str.contains(control)].reset_index(drop=True)
        note = "The literal union has 170 hits, versus 167 printed; High SD2 remains selected."
    else:
        raise ValueError(f"no selected source reader for {pub_id}")
    return RawScreen(rows, t.sources, count_note=note)
