"""Locate user-supplied source files and read documented archive members."""

import gzip
import io
import json
import zipfile
from pathlib import Path
from typing import ClassVar

import pandas as pd


class MissingSource(FileNotFoundError):
    """A required original file or exact archive member is unavailable."""


class Assets:
    """Resolve publication-specific filenames, optionally through a user-supplied path map."""

    def __init__(self, source_root, source_map=None):
        self.root = Path(source_root).resolve()
        self.mapping = json.loads(Path(source_map).read_text()) if source_map else {}
        if not isinstance(self.mapping, dict):
            raise TypeError("source map must be an object keyed by publication ID")

    def mapped(self, pub_id, key):
        entries = self.mapping.get(pub_id, {})
        if not isinstance(entries, dict):
            raise TypeError(f"{pub_id}: source map entry must be an object")
        if key not in entries:
            return None
        path = Path(entries[key])
        path = path if path.is_absolute() else self.root / path
        if not path.is_file():
            raise MissingSource(f"{pub_id}: mapped source is not a regular file: {key}")
        return path

    def find(self, pub_id, name):
        key = Path(name).name
        mapped = self.mapped(pub_id, name)
        if mapped is None and key != name:
            mapped = self.mapped(pub_id, key)
        if mapped is not None:
            return mapped
        path = self.root / pub_id / key
        return path if path.is_file() else None

    def require(self, pub_id, name):
        path = self.find(pub_id, name)
        if path is None:
            raise MissingSource(f"{pub_id}: required source file is missing: {Path(name).name}")
        return path


class Tables:
    """Read the original table layout, supporting full bundles or explicit extracted members."""

    XML_PARTS: ClassVar[dict] = {
        "xl_sharedStrings.xml.gz": "xl/sharedStrings.xml",
        "xl_worksheets_sheet8.xml": "xl/worksheets/sheet8.xml",
        "xl_worksheets_sheet10.xml": "xl/worksheets/sheet10.xml",
    }

    def __init__(self, assets, pub_id):
        self.assets = assets
        self.pub_id = pub_id
        self.sources = []

    def path(self, name):
        self.sources.append(name)
        direct = self.assets.find(self.pub_id, name)
        if direct is not None:
            return direct
        if "#member=" not in name:
            return self.assets.require(self.pub_id, name)
        workbook, alias = name.split("#member=", 1)
        if alias not in self.XML_PARTS:
            raise MissingSource(f"{self.pub_id}: unsupported workbook member: {alias}")
        original = self.assets.require(self.pub_id, workbook)
        with zipfile.ZipFile(original) as archive:
            try:
                value = archive.read(self.XML_PARTS[alias])
            except KeyError as exc:
                raise MissingSource(f"{self.pub_id}: workbook member missing: "
                                    f"{self.XML_PARTS[alias]}") from exc
        # The same XML reader accepts saved member files and streams from the original workbook.
        return io.BytesIO(gzip.compress(value, mtime=0) if alias.endswith(".gz") else value)

    def member(self, outer, *members):
        key = " > ".join((Path(outer).name,) + members)
        direct = self.assets.mapped(self.pub_id, key)
        self.sources.append(key)
        if direct is not None:
            return direct
        value = self.assets.find(self.pub_id, outer)
        if value is None:
            # Supplying the exact final member is an alternative to supplying the outer bundle.
            return self.assets.require(self.pub_id, members[-1])
        for member in members:
            with zipfile.ZipFile(value) as archive:
                try:
                    value = io.BytesIO(archive.read(member))
                except KeyError as exc:
                    raise MissingSource(f"{self.pub_id}: archive member missing: {member}") from exc
        return value

    def excel(self, name, sheet=0, header=0, members=(), need=()):
        source = self.member(name, *members) if members else self.path(name)
        if need:
            frame = pd.read_excel(source, sheet_name=sheet, header=None, dtype=object,
                                  engine="openpyxl")
            for i in range(min(15, len(frame))):
                names = [str(c).strip() if pd.notna(c) else "" for c in frame.iloc[i]]
                if all(c in names for c in need):
                    frame = frame.iloc[i + 1:].copy()
                    frame.columns = names
                    frame = frame.reset_index(drop=True)
                    break
            else:
                raise ValueError(f"header {need} absent from {name}/{sheet}")
        else:
            frame = pd.read_excel(source, sheet_name=sheet, header=header, dtype=object,
                                  engine="openpyxl")
        frame.columns = [str(c).strip() for c in frame.columns]
        return frame

    def csv(self, name, sep=",", members=()):
        source = self.member(name, *members) if members else self.path(name)
        frame = pd.read_csv(source, sep=sep, dtype=object)
        frame.columns = [str(c).strip() for c in frame.columns]
        return frame
