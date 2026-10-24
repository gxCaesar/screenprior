"""Small valid inputs and explicit defects for source and mask reconstruction."""

import csv
import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from gene_mapping import assemble_rows
from prepare_targets import COUNT_FIELDS, check_counts, load_author_blocks, load_config
from readers import raw_rows, read_screen
from source_files import Assets, MissingSource, Tables

HERE = Path(__file__).resolve().parent


def write_rows(path, records):
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def test_mapping_keeps_controls_duplicates_and_unmeasured_genes_distinct():
    """A duplicate non-hit cannot erase a hit; a control or unmeasured gene cannot enter."""
    source = raw_rows(["TCEB2", "ELOB", "CTRL", "NO-TARGET_1", "X"],
                      [True, True, True, True, False], [True, False, False, True, False])
    counts, masks, _, _ = assemble_rows(source, np.array(["CTRL", "ELOB", "X"]),
                                       {"CTRL", "ELOB"}, "example")
    assert counts["raw_scored"] == 3 and counts["mapped_scored"] == 2
    assert masks["tested"].tolist() == [True, True, False]
    assert masks["hit"].tolist() == [False, True, False]
    assert masks["pool"].tolist() == [True, True, False]


def test_supplied_archive_and_extracted_member_match_and_missing_member_refuses(tmp_path):
    """Archive resolution may change the storage route, never the table or missing-member rule."""
    pub = "pmid_example"
    folder = tmp_path / pub
    folder.mkdir()
    inner = io.BytesIO()
    with zipfile.ZipFile(inner, "w") as archive:
        archive.writestr("table.csv", "gene,p\nA,0.01\nB,0.2\n")
    with zipfile.ZipFile(folder / "bundle.zip", "w") as archive:
        archive.writestr("tables.zip", inner.getvalue())
    table = Tables(Assets(tmp_path), pub)
    frame = table.csv("bundle.zip", members=("tables.zip", "table.csv"))
    assert frame.gene.tolist() == ["A", "B"] and frame.p.tolist() == ["0.01", "0.2"]
    with pytest.raises(MissingSource, match="archive member missing"):
        table.member("bundle.zip", "absent.xlsx")
    (folder / "standalone.csv").write_text("gene,p\nA,0.01\nB,0.2\n")
    direct = table.csv("not-supplied.zip", members=("standalone.csv",))
    assert direct.equals(frame)
    with pytest.raises(MissingSource, match="required source file is missing"):
        table.path("absent.xlsx")


def test_explicit_missing_path_does_not_silently_fall_back(tmp_path):
    """A mistaken explicit source map must fail even if a similarly named default file exists."""
    folder = tmp_path / "pmid_example"
    folder.mkdir()
    (folder / "table.csv").write_text("gene,p\nA,0.01\n")
    mapping = tmp_path / "paths.json"
    mapping.write_text(json.dumps({"pmid_example": {"table.csv": "absent.csv"}}))
    with pytest.raises(MissingSource, match="mapped source is not a regular file"):
        Assets(tmp_path, mapping).require("pmid_example", "table.csv")


def test_original_workbook_members_support_the_sparse_xml_reader(tmp_path):
    """The full-workbook route must preserve sparse cell positions and the author hit flag."""
    folder = tmp_path / "pmid_40615675"
    folder.mkdir()
    namespace = 'xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"'
    sheet10 = (f"<worksheet {namespace}><sheetData><row r='1'><c r='A1'/></row>"
               "<row r='2'><c r='A2' t='s'><v>0</v></c>"
               "<c r='B2' t='s'><v>1</v></c></row>"
               "<row r='3'><c r='A3' t='s'><v>2</v></c>"
               "<c r='B3'><v>0.01</v></c></row></sheetData></worksheet>")
    empty_rows = "".join(f"<row r='{i}'><c r='A{i}'/></row>" for i in range(1, 6))
    sheet8 = (f"<worksheet {namespace}><sheetData>" + empty_rows
              + "<row r='6'><c r='B6' t='s'><v>2</v></c>"
              "<c r='C6' t='s'><v>1</v></c><c r='E6'><v>1</v></c></row>"
              "</sheetData></worksheet>")
    with zipfile.ZipFile(folder / "41588_2025_2222_MOESM4_ESM.xlsx", "w") as archive:
        archive.writestr("xl/sharedStrings.xml", f"<sst {namespace}><si><t>Gene</t></si>"
                         "<si><t>921</t></si><si><t>X</t></si></sst>")
        archive.writestr("xl/worksheets/sheet10.xml", sheet10)
        archive.writestr("xl/worksheets/sheet8.xml", sheet8)
    source = read_screen("pmid_40615675", {}, Assets(tmp_path), tmp_path)
    assert source.rows.to_dict("list") == {"raw_id": ["X"], "scored": [True], "hit": [True]}


def test_selected_config_refuses_missing_coverage_and_low_substitution(tmp_path):
    """The40-target interface cannot silently return39targets or substitute the Low screen."""
    config = load_config(HERE / "targets.csv")
    short = tmp_path / "short.csv"
    write_rows(short, config[:-1])
    with pytest.raises(ValueError, match="exactly 40"):
        load_config(short)
    config[-1]["sheet"] = "SD1"
    write_rows(short, config)
    with pytest.raises(ValueError, match="selected High SD2"):
        load_config(short)


def test_count_coverage_accepts_literal170_and_refuses_printed167():
    """A superficially plausible printed hit count must not replace the literal source count."""
    config = load_config(HERE / "targets.csv")[-1]
    counts = {key: int(config[key]) for key in COUNT_FIELDS}
    check_counts(config, counts, [])
    counts["raw_hits"] = 167
    with pytest.raises(ValueError, match="raw_hits: expected 170, observed 167"):
        check_counts(config, counts, [])


def test_missing_author_call_column_and_replicate_row_mismatch_refuse(tmp_path):
    """Required Q columns and aligned two/three-replicate gene identities are load-bearing."""
    pub = "pmid_41285767"
    folder = tmp_path / pub
    folder.mkdir()
    path = folder / "41467_2025_65277_MOESM3_ESM.xlsx"
    frame = pd.DataFrame({"Gene": ["A"], "Enrichment_Qvalue (2 Reps)": [.1],
                          "Enrichment_Qvalue (3 Reps)": [.2], "C3": [0], "C4": [0],
                          "C5": [0], "C6": [0], "Repeated Gene": ["A"]})
    frame.to_excel(path, sheet_name="SD2", index=False, startrow=1)
    source = read_screen(pub, {"sheet": "SD2"}, Assets(tmp_path), tmp_path)
    assert source.rows.hit.tolist() == [True]
    frame["Repeated Gene"] = ["B"]
    frame.to_excel(path, sheet_name="SD2", index=False, startrow=1)
    with pytest.raises(ValueError, match="two gene columns differ"):
        read_screen(pub, {"sheet": "SD2"}, Assets(tmp_path), tmp_path)
    frame.drop(columns=["Enrichment_Qvalue (3 Reps)"]).to_excel(
        path, sheet_name="SD2", index=False, startrow=1)
    with pytest.raises(KeyError, match="Enrichment_Qvalue"):
        read_screen(pub, {"sheet": "SD2"}, Assets(tmp_path), tmp_path)


def test_block_configuration_is_complete_and_contains_only_publication_ids(tmp_path):
    """A missing target or duplicate donor ID cannot silently alter the configured donor set."""
    selected = [row["pub_id"] for row in load_config(HERE / "targets.csv")]
    blocks = load_author_blocks(HERE / "author-blocks.csv", selected)
    assert len(blocks) == 40 and sum(map(len, blocks.values())) == 33
    with open(HERE / "author-blocks.csv", newline="") as fh:
        records = list(csv.DictReader(fh))
    records[0]["blocked_donor_ids"] = '["123","123"]'
    path = tmp_path / "blocks.csv"
    write_rows(path, records)
    with pytest.raises(ValueError, match="duplicate blocked donor"):
        load_author_blocks(path, selected)
    with pytest.raises(ValueError, match="selected target order"):
        load_author_blocks(HERE / "author-blocks.csv", selected[:-1])
