#!/usr/bin/env python3
"""
PDF Editor — Streamlit single-file app with hybrid parsing (rule-based + optional LLM)

What it does
- User uploads a PDF
- App parses the PDF into a "document index" of editable datapoints:
  - Key/value fields (e.g., "Sample ID:", "Strain:", "Collected:", etc.)
  - Table columns (inferred from header rows + column alignment), and their cells
- User types a natural language instruction
- App uses rule-based parsing first (fast, deterministic), with optional LLM fallback
- App fuzzy-matches the user's referenced "title" against extracted titles
- Applies deterministic edits IN-PLACE (redact + insert) to preserve original layout/styling
- User downloads the edited PDF

Supported instruction patterns (examples)
1) Randomize a field/column within a numeric range:
   - "randomize LOD between 10-30%"
   - "randomize loq 0.1-0.4"
   - "randomize total thc % between 20 and 35"
   - '"LOD (mg/g)" change all these values to random numbers within a 5-10% range'
2) Set a field to a specific value:
   - "set Sample ID to 2511HW-WFML-REV2"
   - "change Strain Name to Watermelon Freeze"
3) Shift all dates by N days:
   - "shift all dates by +7 days"
   - "shift dates -3 days"

Parsing Strategy
- Primary: Rule-based parsing (fast, deterministic, works offline)
- Fallback: Optional LLM parsing (requires OpenAI API key, handles complex natural language)
- Enable LLM fallback in the UI for better handling of complex instructions

Notes
- Works best on digitally generated PDFs with selectable text (not scanned images).
- Table inference is heuristic (but good for COA-style aligned tables).
- Fuzzy matching uses rapidfuzz if available; otherwise difflib.

Install
  pip install streamlit pymupdf rapidfuzz python-dateutil
  # Optional for LLM fallback:
  pip install openai

Run
  streamlit run pdf_editor.py
"""

from __future__ import annotations

import io
import json
import math
import random
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union

import fitz  # PyMuPDF
import streamlit as st

try:
    from rapidfuzz import process as rf_process
    from rapidfuzz import fuzz as rf_fuzz

    HAS_RAPIDFUZZ = True
except Exception:
    import difflib

    HAS_RAPIDFUZZ = False


# -----------------------------
# Config
# -----------------------------

REDACT_FILL_RGB = (1, 1, 1)  # white
DEFAULT_MIN_FONT_SIZE = 4.0
DEFAULT_FUZZY_THRESHOLD = 80  # 0-100
AMBIGUITY_DELTA = 4  # if top2 scores within this, treat as ambiguous

DATE_PATTERNS = [
    # MM/DD/YYYY or M/D/YYYY
    re.compile(r"\b(\d{1,2})/(\d{1,2})/(\d{2,4})\b"),
    # YYYY-MM-DD
    re.compile(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b"),
]


# -----------------------------
# Helpers: colors, rects, styles
# -----------------------------

def int_to_rgb_float(color_int: int) -> Tuple[float, float, float]:
    r = (color_int >> 16) & 255
    g = (color_int >> 8) & 255
    b = color_int & 255
    return (r / 255.0, g / 255.0, b / 255.0)


def rect_area(r: fitz.Rect) -> float:
    if r.is_empty:
        return 0.0
    return float(r.width * r.height)


def rect_union(rects: List[fitz.Rect]) -> fitz.Rect:
    if not rects:
        return fitz.Rect(0, 0, 0, 0)
    out = rects[0]
    for r in rects[1:]:
        out = out | r
    return out


def rect_overlap_area(a: fitz.Rect, b: fitz.Rect) -> float:
    inter = a & b
    if inter.is_empty:
        return 0.0
    return float(inter.width * inter.height)


def rect_center(r: fitz.Rect) -> Tuple[float, float]:
    return ((r.x0 + r.x1) / 2.0, (r.y0 + r.y1) / 2.0)


@dataclass
class SpanStyle:
    font: str
    size: float
    color_rgb: Tuple[float, float, float]


def extract_spans(page: fitz.Page) -> List[Dict[str, Any]]:
    """
    Flat list of text spans with bbox/font/size/color.
    """
    d = page.get_text("dict")
    spans: List[Dict[str, Any]] = []
    for block in d.get("blocks", []):
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                text = span.get("text", "")
                if not text or not text.strip():
                    continue
                spans.append(
                    {
                        "text": text,
                        "bbox": fitz.Rect(span["bbox"]),
                        "font": span.get("font", "Helvetica"),
                        "size": float(span.get("size", 10.0)),
                        "color": int(span.get("color", 0)),
                    }
                )
    return spans


def group_spans_into_lines(spans: List[Dict[str, Any]], y_tol: float = 2.5) -> List[Dict[str, Any]]:
    """
    Group spans into lines by y-center proximity.
    Returns list of line dicts with bbox, spans, text.
    """
    if not spans:
        return []

    spans_sorted = sorted(spans, key=lambda s: (s["bbox"].y0, s["bbox"].x0))
    lines: List[List[Dict[str, Any]]] = []

    for sp in spans_sorted:
        y_center = (sp["bbox"].y0 + sp["bbox"].y1) / 2.0
        placed = False
        for ln in lines:
            ref = ln[0]
            ref_y = (ref["bbox"].y0 + ref["bbox"].y1) / 2.0
            if abs(y_center - ref_y) <= y_tol:
                ln.append(sp)
                placed = True
                break
        if not placed:
            lines.append([sp])

    out_lines: List[Dict[str, Any]] = []
    for i, ln in enumerate(lines):
        ln_sorted = sorted(ln, key=lambda s: s["bbox"].x0)
        bbox = rect_union([s["bbox"] for s in ln_sorted])
        text = "".join([s["text"] for s in ln_sorted]).strip()
        out_lines.append({"line_id": i, "bbox": bbox, "spans": ln_sorted, "text": text})

    out_lines.sort(key=lambda l: (l["bbox"].y0, l["bbox"].x0))
    for i, l in enumerate(out_lines):
        l["line_id"] = i
    return out_lines


def pick_style_near_rect(spans: List[Dict[str, Any]], target: fitz.Rect) -> SpanStyle:
    """
    Pick style from a span overlapping target; else nearest span by center distance.
    """
    best = None
    best_overlap = 0.0
    for s in spans:
        ov = rect_overlap_area(s["bbox"], target)
        if ov > best_overlap:
            best_overlap = ov
            best = s
    if best is None:
        tx, ty = rect_center(target)
        best_d2 = None
        for s in spans:
            sx, sy = rect_center(s["bbox"])
            d2 = (sx - tx) ** 2 + (sy - ty) ** 2
            if best_d2 is None or d2 < best_d2:
                best_d2 = d2
                best = s

    if best is None:
        return SpanStyle(font="Helvetica", size=10.0, color_rgb=(0, 0, 0))

    return SpanStyle(
        font=best["font"],
        size=float(best["size"]),
        color_rgb=int_to_rgb_float(int(best["color"])),
    )


def insert_text_fit(
    page: fitz.Page,
    rect: fitz.Rect,
    text: str,
    style: SpanStyle,
    align: int = fitz.TEXT_ALIGN_LEFT,
    min_font_size: float = DEFAULT_MIN_FONT_SIZE,
) -> None:
    """
    Insert text into rect; shrink font until it fits.
    """
    fontsize = float(style.size)
    while fontsize >= min_font_size:
        rc = page.insert_textbox(
            rect,
            text,
            fontname=style.font,
            fontsize=fontsize,
            color=style.color_rgb,
            align=align,
        )
        if rc >= 0:
            return
        fontsize -= 0.25
    page.insert_textbox(
        rect,
        text,
        fontname=style.font,
        fontsize=min_font_size,
        color=style.color_rgb,
        align=align,
    )


# -----------------------------
# Document Index structures
# -----------------------------

@dataclass
class Target:
    page_index: int
    rect: fitz.Rect
    style: SpanStyle
    original_text: str
    kind: str  # "field_value" | "table_cell" | "date_span"


@dataclass
class TitleEntry:
    title: str
    normalized: str
    targets: List[Target]
    meta: Dict[str, Any]


def normalize_title(s: str) -> str:
    s0 = s.lower().strip()
    s0 = re.sub(r"[\u200b-\u200f]", "", s0)
    s0 = re.sub(r"[\(\)\[\]\{\}]", " ", s0)
    s0 = re.sub(r"[^a-z0-9%/.\- +]", " ", s0)
    s0 = re.sub(r"\s+", " ", s0).strip()
    # remove trailing colon
    s0 = s0[:-1].strip() if s0.endswith(":") else s0
    # collapse units a bit
    s0 = s0.replace("mg / g", "mg/g").replace("mg g", "mg/g").replace("mg per g", "mg/g")
    return s0


def normalize_cannabinoid_name(name: str) -> Optional[str]:
    """
    Normalize cannabinoid names to standard forms.
    Returns None if not a recognized cannabinoid.
    """
    name_lower = name.lower().strip()
    
    # THCa variations
    if re.search(r'\bthc[-\s]?a\b', name_lower) or name_lower == 'thca':
        return "THCa %"
    
    # Delta-9 THC variations
    if re.search(r'\b(?:delta[-\s]?9|delta\s+9|d9|Δ9|Δ\s*9)\b', name_lower, re.IGNORECASE):
        return "Δ9 %"
    
    # Total THC
    if re.search(r'\btotal\s+thc\b', name_lower):
        return "Total THC %"
    
    # Total Cannabinoids
    if re.search(r'\btotal\s+cannabinoids?\b', name_lower):
        return "Total Cannabinoids %"
    
    # CBD
    if re.search(r'\bcbd\b', name_lower):
        return "CBD %"
    
    return None


def build_aliases(title: str) -> List[str]:
    """
    Rule-based aliases (still generic, not hardcoded scenarios):
    - remove units in parentheses
    - remove percent sign
    - remove punctuation
    - keep "short forms" for headers like "lod (mg/g)" -> "lod"
    """
    base = title.strip()
    aliases = set()

    aliases.add(base)
    # remove colon
    aliases.add(base.rstrip(":").strip())

    # remove parens content
    aliases.add(re.sub(r"\(.*?\)", "", base).strip())

    # remove % sign
    aliases.add(base.replace("%", "").strip())

    # if has parentheses, also keep inside
    m = re.findall(r"\((.*?)\)", base)
    for inner in m:
        if inner.strip():
            aliases.add(inner.strip())

    # if short token-like header, keep first token
    tokens = re.split(r"\s+", re.sub(r"[^\w%/.-]+", " ", base).strip())
    if tokens:
        aliases.add(tokens[0])

    # normalized forms
    out = []
    for a in aliases:
        n = normalize_title(a)
        if n:
            out.append(n)
    # de-dupe
    return sorted(set(out), key=len, reverse=True)


# -----------------------------
# Table inference (heuristic)
# -----------------------------

def cluster_x_positions(xs: List[float], tol: float = 12.0) -> List[float]:
    if not xs:
        return []
    xs_sorted = sorted(xs)
    clusters = [[xs_sorted[0]]]
    for x in xs_sorted[1:]:
        if abs(x - clusters[-1][-1]) <= tol:
            clusters[-1].append(x)
        else:
            clusters.append([x])
    return [sum(c) / len(c) for c in clusters]


def infer_table_runs(lines: List[Dict[str, Any]], page_width: float) -> List[Tuple[int, int]]:
    """
    Find consecutive runs of "cell-ish" lines (multiple spans spread wide).
    """
    feats = []
    for ln in lines:
        spans = ln["spans"]
        if len(spans) < 3:
            feats.append(False)
            continue
        x0s = [s["bbox"].x0 for s in spans]
        spread = max(x0s) - min(x0s)
        feats.append(spread >= (0.35 * page_width))

    runs: List[Tuple[int, int]] = []
    i = 0
    while i < len(feats):
        if not feats[i]:
            i += 1
            continue
        start = i
        while i < len(feats) and feats[i]:
            i += 1
        end = i - 1
        if end - start + 1 >= 4:
            runs.append((start, end))
    return runs


def build_table_from_run(
    run_lines: List[Dict[str, Any]],
    page_width: float,
) -> Dict[str, Any]:
    """
    Build a table model from a run of lines by clustering x positions.
    """
    all_x = []
    for ln in run_lines:
        for sp in ln["spans"]:
            all_x.append(sp["bbox"].x0)

    anchors = sorted(cluster_x_positions(all_x, tol=12.0))
    if len(anchors) < 2:
        return {}

    # build column bounds using midpoints
    bounds: List[Tuple[float, float]] = []
    for i, a in enumerate(anchors):
        if i + 1 < len(anchors):
            mid = (anchors[i] + anchors[i + 1]) / 2.0
            bounds.append((a - 1.0, mid))
        else:
            bounds.append((a - 1.0, page_width + 1.0))

    def assign_col(x0: float) -> Optional[int]:
        for ci, (bx0, bx1) in enumerate(bounds):
            if x0 >= bx0 and x0 < bx1:
                return ci
        return None

    # rows & cells
    rows: List[Dict[str, Any]] = []
    for ri, ln in enumerate(run_lines):
        cells: Dict[int, Dict[str, Any]] = {}
        for sp in sorted(ln["spans"], key=lambda s: s["bbox"].x0):
            ci = assign_col(sp["bbox"].x0)
            if ci is None:
                continue
            if ci not in cells:
                cells[ci] = {
                    "col_id": ci,
                    "text": sp["text"].strip(),
                    "bbox": sp["bbox"],
                    "spans": [sp],
                }
            else:
                cells[ci]["text"] = (cells[ci]["text"] + " " + sp["text"].strip()).strip()
                cells[ci]["bbox"] = cells[ci]["bbox"] | sp["bbox"]
                cells[ci]["spans"].append(sp)

        row_bbox = ln["bbox"]
        rows.append(
            {
                "row_id": ri,
                "bbox": row_bbox,
                "cells": [cells[k] for k in sorted(cells.keys())],
            }
        )

    # choose header row among first 3 by average font size
    def avg_font_size(row: Dict[str, Any]) -> float:
        sizes = []
        for cell in row["cells"]:
            for sp in cell["spans"]:
                sizes.append(float(sp.get("size", 0.0)))
        return sum(sizes) / len(sizes) if sizes else 0.0

    header_idx = 0
    best = -1.0
    for idx in range(min(3, len(rows))):
        a = avg_font_size(rows[idx])
        if a > best:
            best = a
            header_idx = idx

    for i, r in enumerate(rows):
        r["is_header"] = (i == header_idx)

    headers_by_col: Dict[int, str] = {}
    for cell in rows[header_idx]["cells"]:
        headers_by_col[int(cell["col_id"])] = cell["text"].strip()

    columns = []
    for ci, (bx0, bx1) in enumerate(bounds):
        columns.append(
            {
                "col_id": ci,
                "x0": float(bx0),
                "x1": float(bx1),
                "header_text": headers_by_col.get(ci, "").strip(),
            }
        )

    tbx0 = min(r["bbox"].x0 for r in rows)
    tby0 = min(r["bbox"].y0 for r in rows)
    tbx1 = max(r["bbox"].x1 for r in rows)
    tby1 = max(r["bbox"].y1 for r in rows)

    return {
        "bbox": fitz.Rect(tbx0, tby0, tbx1, tby1),
        "header_row_id": header_idx,
        "columns": columns,
        "rows": rows,
    }


# -----------------------------
# Index builder
# -----------------------------

FIELD_LABEL_RE = re.compile(r"^\s*([A-Za-z][A-Za-z0-9 \-/#()%]+?):\s*(.*)\s*$")


def build_document_index(doc: fitz.Document) -> Tuple[List[TitleEntry], Dict[str, Any]]:
    """
    Build an index of titles -> targets.
    Titles come from:
      - field labels in "Label: Value" lines
      - table column headers
    Also builds a metadata dict (debug).
    """
    entries: Dict[str, TitleEntry] = {}
    debug: Dict[str, Any] = {"pages": []}

    for pno in range(len(doc)):
        page = doc[pno]
        spans = extract_spans(page)
        lines = group_spans_into_lines(spans, y_tol=2.5)
        page_w = float(page.rect.width)

        page_debug = {"page": pno + 1, "field_candidates": [], "tables": []}

        # 1) Field labels: detect "Label: Value" from line text, then locate value rect
        for ln in lines:
            m = FIELD_LABEL_RE.match(ln["text"])
            if not m:
                continue
            label = m.group(1).strip()
            value_text_inline = m.group(2).strip()

            # locate label span(s) on that line (best effort: spans containing label or starting with it)
            label_spans = []
            for sp in ln["spans"]:
                if sp["text"].strip().startswith(label) or label in sp["text"]:
                    label_spans.append(sp)
            if not label_spans:
                continue

            label_rect = rect_union([s["bbox"] for s in label_spans])

            # value spans: spans on same line whose x0 > label_rect.x1 + small gap
            value_spans = [sp for sp in ln["spans"] if sp["bbox"].x0 > (label_rect.x1 + 2.0)]
            # if none, fallback: inline value may be in same span after colon; try find colon span
            if not value_spans and value_text_inline:
                # try find span that contains colon and value
                for sp in ln["spans"]:
                    if ":" in sp["text"]:
                        # approximate: value rect = from colon to end of that span
                        value_spans = [sp]
                        break

            if not value_spans:
                continue

            value_rect = rect_union([s["bbox"] for s in value_spans])
            style = pick_style_near_rect(spans, value_rect)

            key_norm = normalize_title(label)
            aliases = build_aliases(label)

            page_debug["field_candidates"].append(
                {"label": label, "value_preview": value_text_inline, "value_bbox": [value_rect.x0, value_rect.y0, value_rect.x1, value_rect.y1]}
            )

            for alias_norm in aliases:
                k = alias_norm
                if k not in entries:
                    entries[k] = TitleEntry(
                        title=label,
                        normalized=k,
                        targets=[],
                        meta={"type": "field", "aliases": aliases},
                    )
                entries[k].targets.append(
                    Target(
                        page_index=pno,
                        rect=value_rect,
                        style=style,
                        original_text=value_text_inline,
                        kind="field_value",
                    )
                )

        # 2) Table inference and headers -> column targets
        runs = infer_table_runs(lines, page_w)
        tables = []
        for (sidx, eidx) in runs:
            run_lines = lines[sidx : eidx + 1]
            table = build_table_from_run(run_lines, page_w)
            if not table or "columns" not in table:
                continue

            # basic sanity: need at least 3 columns with non-empty header
            headers = [c["header_text"] for c in table["columns"] if c["header_text"]]
            if len(headers) < 2:
                continue

            tables.append(table)

        for tidx, table in enumerate(tables):
            headers = [c["header_text"] for c in table["columns"]]
            page_debug["tables"].append(
                {
                    "table_id": tidx,
                    "bbox": [table["bbox"].x0, table["bbox"].y0, table["bbox"].x1, table["bbox"].y1],
                    "headers": headers,
                }
            )

            # Build targets for each column header -> all cells beneath it
            header_row_id = int(table["header_row_id"])
            for col in table["columns"]:
                header_text = (col.get("header_text") or "").strip()
                if not header_text:
                    continue

                aliases = build_aliases(header_text)
                # collect all cells in that col excluding header row
                col_id = int(col["col_id"])
                col_targets: List[Target] = []

                for row in table["rows"]:
                    if int(row["row_id"]) == header_row_id:
                        continue
                    # find cell in that row with col_id
                    for cell in row["cells"]:
                        if int(cell["col_id"]) != col_id:
                            continue
                        cell_text = cell["text"].strip()
                        # skip empty
                        if not cell_text:
                            continue
                        cell_rect: fitz.Rect = cell["bbox"]
                        style = pick_style_near_rect(spans, cell_rect)
                        col_targets.append(
                            Target(
                                page_index=pno,
                                rect=cell_rect,
                                style=style,
                                original_text=cell_text,
                                kind="table_cell",
                            )
                        )

                for alias_norm in aliases:
                    k = alias_norm
                    if k not in entries:
                        entries[k] = TitleEntry(
                            title=header_text,
                            normalized=k,
                            targets=[],
                            meta={"type": "table_column", "aliases": aliases, "table_id": tidx, "col_id": col_id, "headers": headers},
                        )
                    # append all targets (may include duplicates across aliases — okay)
                    entries[k].targets.extend(col_targets)

        # 3) Dates: index all date-like spans as "date" title group
        date_targets: List[Target] = []
        for sp in spans:
            t = sp["text"].strip()
            if not t:
                continue
            if is_date_like(t):
                style = pick_style_near_rect(spans, sp["bbox"])
                date_targets.append(Target(pno, sp["bbox"], style, t, "date_span"))

        if date_targets:
            # generic date group
            for key in ["date", "dates", "all dates"]:
                k = normalize_title(key)
                if k not in entries:
                    entries[k] = TitleEntry(title="Dates", normalized=k, targets=[], meta={"type": "date_group", "aliases": [k]})
                entries[k].targets.extend(date_targets)

        # 4) Extract cannabinoid values (THCa %, Δ9 %, Total THC %, Total Cannabinoids, etc.)
        # Enhanced patterns to catch more variations
        cannabinoid_patterns = [
            # Pattern: "THCa: 25.5%" or "THCa 25.5%" or "25.5% THCa"
            r'\b(THCa?|THC-A|THC\s*A|THC\s+a)\s*[:\s]*([\d.]+)\s*%',
            r'\b([\d.]+)\s*%\s*(THCa?|THC-A|THC\s*A|THC\s+a)\b',
            # Pattern: "Delta-9: 0.8%" or "Δ9: 0.8%" or "D9: 0.8%"
            r'\b(Delta-?9|Delta\s+9|Δ9|Δ\s*9|D9|Delta\s*9)\s*[:\s]*([\d.]+)\s*%',
            r'\b([\d.]+)\s*%\s*(Delta-?9|Delta\s+9|Δ9|Δ\s*9|D9)\b',
            # Pattern: "Total THC: 22.5%" or "Total THC 22.5%"
            r'\b(Total\s+THC|Total\s*THC)\s*[:\s]*([\d.]+)\s*%',
            r'\b([\d.]+)\s*%\s*(Total\s+THC|Total\s*THC)\b',
            # Pattern: "Total Cannabinoids: 25.0%" or "Total Cannabinoids 25.0%"
            r'\b(Total\s+Cannabinoids?|Total\s*Cannabinoids?)\s*[:\s]*([\d.]+)\s*%',
            r'\b([\d.]+)\s*%\s*(Total\s+Cannabinoids?|Total\s*Cannabinoids?)\b',
            # Also check table cells and standalone values
            r'\b(CBD|CBN|CBG)\s*[:\s]*([\d.]+)\s*%',
        ]
        
        cannabinoid_targets: Dict[str, List[Target]] = {}
        
        # First pass: Check individual spans
        for sp in spans:
            t = sp["text"].strip()
            if not t:
                continue
            
            # Check for cannabinoid patterns
            for pattern in cannabinoid_patterns:
                matches = re.finditer(pattern, t, re.IGNORECASE)
                for match in matches:
                    # Extract cannabinoid name and value
                    if len(match.groups()) >= 2:
                        name_part = match.group(1).strip()
                        value_part = match.group(2).strip()
                        
                        # Normalize name
                        name_norm = normalize_cannabinoid_name(name_part)
                        if name_norm:
                            style = pick_style_near_rect(spans, sp["bbox"])
                            if name_norm not in cannabinoid_targets:
                                cannabinoid_targets[name_norm] = []
                            cannabinoid_targets[name_norm].append(
                                Target(pno, sp["bbox"], style, t, "cannabinoid_value")
                            )
        
        # Second pass: Check table cells for cannabinoid values (already processed tables)
        # Look for numeric values near cannabinoid labels in tables
        # Note: tables variable is defined earlier in the function scope
        for table in tables:
            if "columns" not in table:
                continue
            headers = [c["header_text"] for c in table["columns"]]
            for col_idx, col in enumerate(table["columns"]):
                header_text = col.get("header_text", "").strip()
                if not header_text:
                    continue
                
                # Check if header matches cannabinoid pattern
                name_norm = normalize_cannabinoid_name(header_text)
                if name_norm:
                    # Collect all cells in this column
                    col_id = int(col["col_id"])
                    header_row_id = int(table.get("header_row_id", 0))
                    for row in table.get("rows", []):
                        if int(row.get("row_id", -1)) == header_row_id:
                            continue  # Skip header row
                        for cell in row.get("cells", []):
                            if int(cell.get("col_id", -1)) == col_id:
                                cell_text = cell.get("text", "").strip()
                                # Check if cell contains a percentage value
                                if cell_text and re.search(r'\d+\.?\d*\s*%', cell_text):
                                    if name_norm not in cannabinoid_targets:
                                        cannabinoid_targets[name_norm] = []
                                    cell_rect = cell.get("bbox")
                                    if cell_rect:
                                        style = pick_style_near_rect(spans, cell_rect)
                                        cannabinoid_targets[name_norm].append(
                                            Target(pno, cell_rect, style, cell_text, "cannabinoid_value")
                                        )
        
        # Ensure the 4 priority cannabinoids always exist (even if empty)
        priority_cannabinoids = ["THCa %", "Δ9 %", "Total THC %", "Total Cannabinoids %"]
        for priority_name in priority_cannabinoids:
            if priority_name not in cannabinoid_targets:
                cannabinoid_targets[priority_name] = []
        
        # Add cannabinoid entries
        for name_norm, targets_list in cannabinoid_targets.items():
            if name_norm not in entries:
                entries[name_norm] = TitleEntry(
                    title=name_norm,
                    normalized=name_norm,
                    targets=[],
                    meta={"type": "cannabinoid", "aliases": [name_norm], "is_priority": name_norm in priority_cannabinoids}
                )
            entries[name_norm].targets.extend(targets_list)

        debug["pages"].append(page_debug)

    # Deduplicate targets in each entry by (page, rect coords, original_text)
    def target_key(t: Target) -> Tuple[int, float, float, float, float, str, str]:
        r = t.rect
        return (t.page_index, round(r.x0, 2), round(r.y0, 2), round(r.x1, 2), round(r.y1, 2), t.original_text, t.kind)

    for k, e in entries.items():
        seen = set()
        uniq = []
        for t in e.targets:
            tk = target_key(t)
            if tk in seen:
                continue
            seen.add(tk)
            uniq.append(t)
        e.targets = uniq

    # Return list sorted by title length for nicer UI
    out = sorted(entries.values(), key=lambda e: (e.meta.get("type", ""), len(e.title)))
    return out, debug


# -----------------------------
# Instruction parsing (no LLM)
# -----------------------------

def is_date_like(s: str) -> bool:
    s = s.strip()
    for pat in DATE_PATTERNS:
        if pat.search(s):
            return True
    return False


def parse_shift_days(instr: str) -> Optional[int]:
    # "shift all dates by +7 days" or "shift dates -3"
    m = re.search(r"\bshift\b.*\bdates?\b.*\bby\b\s*([+-]?\d+)\s*(day|days)?\b", instr, re.IGNORECASE)
    if not m:
        m = re.search(r"\bshift\b.*\bdates?\b\s*([+-]?\d+)\s*(day|days)?\b", instr, re.IGNORECASE)
    if m:
        return int(m.group(1))
    return None


def parse_formula_edit(instr: str) -> Optional[Dict[str, Any]]:
    """
    Parse formula-based edits like "Ensure THCa × 0.877 + Δ9 = Total THC"
    """
    # THCa formula: THCa × 0.877 + Δ9 = Total THC
    formula_match = re.search(
        r'\b(?:ensure|make|calculate|set)\s+(?:that\s+)?(?:THCa|THC-A|THC\s*A)\s*[×x*]\s*0\.877\s*\+\s*(?:Δ9|Delta-?9|D9|Delta\s+9)\s*=\s*(?:Total\s+THC|Total\s+THC\s*%)',
        instr,
        re.IGNORECASE
    )
    if formula_match:
        return {
            "op": "formula_thc",
            "formula": "THCa × 0.877 + Δ9 = Total THC",
            "description": "Calculate Total THC from THCa and Δ9 using conversion factor"
        }
    
    # Reverse-engineer totals
    if re.search(r'\b(?:reverse[-\s]?engineer|make\s+totals?\s+add\s+up|ensure\s+totals?\s+add\s+up)', instr, re.IGNORECASE):
        return {
            "op": "reverse_engineer_totals",
            "description": "Reverse-engineer component values so totals add up correctly"
        }
    
    return None


def parse_conversion_edit(instr: str) -> Optional[Dict[str, Any]]:
    """
    Parse % ↔ mg/g conversion edits
    """
    if re.search(r'\b(?:ensure|make|convert|fix)\s+(?:that\s+)?(?:%|percent|percentage)\s*[↔<->]\s*(?:mg/g|mg\s+per\s+g|mg\s*/\s*g)', instr, re.IGNORECASE):
        return {
            "op": "fix_conversions",
            "description": "Ensure % ↔ mg/g conversions are accurate"
        }
    
    return None


def parse_tier_adjustment(instr: str) -> Optional[Dict[str, Any]]:
    """
    Parse tier-based adjustments (low/mid/high)
    """
    tier_match = re.search(
        r'\b(?:adjust|set|change)\s+(?:values?|numbers?)\s+(?:based\s+on|to|for)\s+(?:tier\s+)?(low|mid|medium|high)\b',
        instr,
        re.IGNORECASE
    )
    if tier_match:
        tier = tier_match.group(1).lower()
        if tier == "medium":
            tier = "mid"
        return {
            "op": "tier_adjustment",
            "tier": tier,
            "description": f"Adjust values based on {tier} tier range"
        }
    
    return None


def parse_randomize(instr: str) -> Optional[Dict[str, Any]]:
    """
    Handles various randomize patterns:
    - "randomize <title> between A-B%"
    - "randomize <title> A-B"
    - "<title>" change all these values to random numbers within A-B% range
    - change <title> to random numbers within A-B% range
    - <title> randomize A-B%
    """
    # First, try to find a numeric range pattern (A-B or A to B)
    range_patterns = [
        # "5-10%", "5 to 10%", "5 and 10%", "between 5 and 10%", "within 5-10%"
        r"(?P<a>\d+(\.\d+)?)\s*[-–—]\s*(?P<b>\d+(\.\d+)?)\s*(?P<pct>%?)",
        r"(?P<a>\d+(\.\d+)?)\s+(?:to|and)\s+(?P<b>\d+(\.\d+)?)\s*(?P<pct>%?)",
        r"(?:between|within|from)\s+(?P<a>\d+(\.\d+)?)\s+(?:and|to|-)\s+(?P<b>\d+(\.\d+)?)\s*(?P<pct>%?)",
    ]
    
    range_match = None
    for pattern in range_patterns:
        range_match = re.search(pattern, instr, re.IGNORECASE)
        if range_match:
            break
    
    if not range_match:
        return None
    
    # Extract range values
    a = float(range_match.group("a"))
    b = float(range_match.group("b"))
    pct_str = range_match.group("pct") or ""
    pct = "%" in pct_str or "%" in instr.lower()
    
    if a > b:
        a, b = b, a
    
    # Check if this is a randomize operation (look for keywords)
    random_keywords = r"\b(random|randomize|randomise|randomly|random numbers)\b"
    if not re.search(random_keywords, instr, re.IGNORECASE):
        return None
    
    # Extract title - try multiple patterns
    title = None
    
    # Pattern 1: Title in quotes at the start: "LOD (mg/g)" change...
    quote_match = re.match(r'^(["\'])(.+?)\1', instr)
    if quote_match:
        title = quote_match.group(2).strip()
    
    # Pattern 2: Title before "change/set/update" at start: "LOD (mg/g)" change... or LOD change...
    if not title:
        # Try to find title before "randomize" or "change" at the start
        before_op = re.search(r'^(.+?)\s+\b(randomize|randomise|change|set|update|make)\b', instr, re.IGNORECASE)
        if before_op:
            potential_title = before_op.group(1).strip().strip('"').strip("'")
            # Skip common prefixes and empty strings
            if potential_title and not re.match(r'^(all|these|the|values?|numbers?)$', potential_title, re.IGNORECASE):
                # Remove any trailing punctuation that might be part of the title
                potential_title = re.sub(r'[:;]$', '', potential_title).strip()
                if potential_title:
                    title = potential_title
    
    # Pattern 3: "randomize <title>" or "change <title> to random"
    if not title:
        after_op = re.search(r'\b(randomize|randomise|change|set|update)\s+(["\']?)(.+?)\2\s+(?:to|all|these|values)', instr, re.IGNORECASE)
        if after_op:
            potential_title = after_op.group(3).strip().strip('"').strip("'")
            # Remove common suffixes
            potential_title = re.sub(r'\s+(all|these|values?|numbers?)$', '', potential_title, flags=re.IGNORECASE).strip()
            if potential_title:
                title = potential_title
    
    # Pattern 4: Extract title between quotes anywhere
    if not title:
        quote_match = re.search(r'(["\'])(.+?)\1', instr)
        if quote_match:
            potential_title = quote_match.group(2).strip()
            if potential_title and len(potential_title) > 2:
                title = potential_title
    
    # Pattern 5: Title before range: "LOD 5-10%" or "LOD between 5-10%" or "LOD (mg/g)" change... to random... within 5-10%
    if not title:
        before_range = instr[:range_match.start()].strip()
        if before_range:
            # First, try to extract quoted title if present
            quote_in_before = re.search(r'(["\'])(.+?)\1', before_range)
            if quote_in_before:
                potential_title = quote_in_before.group(2).strip()
                if potential_title and len(potential_title) > 1:
                    title = potential_title
            else:
                # Remove common operation words and phrases
                # Pattern: "LOD (mg/g)" change all these values to random numbers within a
                # We want to extract "LOD (mg/g)" - but quotes were already handled, so look for text before "change"
                # Try to find text before operation words
                before_op_match = re.search(r'^(.+?)\s+\b(change|set|update|randomize|randomise|make)\b', before_range, re.IGNORECASE)
                if before_op_match:
                    potential_title = before_op_match.group(1).strip().strip('"').strip("'")
                    if potential_title and len(potential_title) > 1:
                        title = potential_title
                else:
                    # No operation word found, try cleaning up the whole thing
                    before_range = re.sub(r'\s+(to|all|these|values?|numbers?|within|between|from|a|an)\s*$', '', before_range, flags=re.IGNORECASE)
                    before_range = before_range.strip().strip('"').strip("'")
                    # Remove trailing colons/semicolons
                    before_range = re.sub(r'[:;]$', '', before_range).strip()
                    if before_range and len(before_range) > 1:
                        title = before_range
    
    if not title:
        return None
    
    # Clean up title
    title = title.strip().strip('"').strip("'")
    # Remove trailing "change", "set", etc if accidentally captured
    title = re.sub(r'\s+(change|set|update|to|all|these|values?|numbers?)$', '', title, flags=re.IGNORECASE).strip()
    
    if not title:
        return None
    
    # Determine decimals
    dec_a = decimals_in_str(range_match.group("a"))
    dec_b = decimals_in_str(range_match.group("b"))
    decimals = max(dec_a, dec_b)
    if decimals == 0:
        decimals = 2 if pct else 3
    
    return {"op": "randomize", "title": title, "min": a, "max": b, "percent": pct, "decimals": decimals}


def decimals_in_str(num_str: str) -> int:
    if "." in num_str:
        return len(num_str.split(".", 1)[1])
    return 0


def parse_set(instr: str) -> Optional[Dict[str, Any]]:
    """
    Handles set operations: "set X to Y" / "change X to Y" / "X change to Y"
    Excludes randomize operations (those should be caught by parse_randomize first)
    """
    # Skip if this looks like a randomize operation
    if re.search(r"\b(random|randomize|randomise|randomly|random numbers)\b", instr, re.IGNORECASE):
        return None
    
    # Skip if contains range patterns (A-B, A to B, etc.)
    if re.search(r"\d+\s*[-–—]\s*\d+|\d+\s+(?:to|and)\s+\d+", instr, re.IGNORECASE):
        return None
    
    # Try pattern: "title" change/set/update to value
    m = re.search(r"([\"']?)(?P<title>.+?)\1\s+\b(set|change|update)\b\s+\bto\b\s+(?P<value>.+)$", instr, re.IGNORECASE)
    if not m:
        # Try pattern: change/set/update "title" to value
        m = re.search(r"\b(set|change|update)\b\s+([\"']?)(?P<title>.+?)\2\s+\bto\b\s+(?P<value>.+)$", instr, re.IGNORECASE)
    if not m:
        # Try pattern: change/set/update title to value (no quotes)
        m = re.search(r"\b(set|change|update)\b\s+(?P<title>.+?)\s+\bto\b\s+(?P<value>.+)$", instr, re.IGNORECASE)
    if not m:
        return None
    
    title = m.group("title").strip().strip('"').strip("'")
    value = m.group("value").strip().strip('"').strip("'")
    
    # Clean up title - remove common suffixes
    title = re.sub(r'\s+(all|these|values?|numbers?)$', '', title, flags=re.IGNORECASE).strip()
    
    return {"op": "set", "title": title, "value": value}


def parse_instruction_with_llm(instr: str, available_titles: List[str] = None) -> Optional[Dict[str, Any]]:
    """
    Optional LLM-based instruction parser (fallback).
    Requires openai package: pip install openai
    API key can be set via:
    - Streamlit secrets: st.secrets["openai"]["api_key"]
    - Environment variable: OPENAI_API_KEY
    """
    try:
        import os
        from openai import OpenAI
        
        # Try Streamlit secrets first (more secure for Streamlit apps)
        api_key = None
        try:
            if hasattr(st, 'secrets') and 'openai' in st.secrets and 'api_key' in st.secrets['openai']:
                api_key = st.secrets['openai']['api_key']
        except Exception:
            pass
        
        # Fallback to environment variable
        if not api_key:
            api_key = os.getenv("OPENAI_API_KEY")
        
        if not api_key:
            return None
        
        client = OpenAI(api_key=api_key)
        
        # Build context about available titles
        titles_context = ""
        if available_titles:
            titles_context = f"\n\nAvailable field/column titles in the PDF:\n" + "\n".join(f"- {t}" for t in available_titles[:50])
        
        prompt = f"""Parse this PDF editing instruction into structured JSON. If multiple edits are requested (separated by "and", commas, etc.), return a JSON array.

Instruction: "{instr}"
{titles_context}

Return JSON with one of these structures:
1. Single operation: {{"op": "randomize", "title": "field_name", "min": 5.0, "max": 10.0, "percent": true, "decimals": 2}}
2. Multiple operations: [{{"op": "set", "title": "Strain", "value": "GG4"}}, {{"op": "randomize", "title": "LOD", "min": 5.0, "max": 10.0, "percent": true}}]

Operation types:
- Randomize: {{"op": "randomize", "title": "field_name", "min": 5.0, "max": 10.0, "percent": true, "decimals": 2}}
- Set value: {{"op": "set", "title": "field_name", "value": "new_value"}}
- Shift dates: {{"op": "shift_dates", "days": 7}}
- Formula edit: {{"op": "formula_thc", "formula": "THCa × 0.877 + Δ9 = Total THC"}}
- Reverse engineer: {{"op": "reverse_engineer_totals"}}
- Fix conversions: {{"op": "fix_conversions"}}
- Tier adjustment: {{"op": "tier_adjustment", "tier": "low|mid|high"}}

Domain-specific operations (for cannabis lab reports):
- "Change THCa %, Δ9 %, Total THC %, Total Cannabinoids" → set operations for each
- "Ensure THCa × 0.877 + Δ9 = Total THC" → formula_thc operation
- "Reverse-engineer numbers so totals add up correctly" → reverse_engineer_totals
- "Ensure % ↔ mg/g conversions are accurate" → fix_conversions
- "Adjust values based on tier ranges (low/mid/high)" → tier_adjustment

Rules:
- Extract the exact field/column title from available titles (use fuzzy matching if needed)
- For randomize: extract min/max range, detect if percentage, preserve decimal precision
- For set: extract the exact value to set
- For dates: extract number of days (positive/negative)
- For cannabinoid operations: recognize THCa, Δ9, Total THC, Total Cannabinoids
- If multiple edits: return array of operations

Return ONLY valid JSON, no explanation."""

        response = client.chat.completions.create(
            model="gpt-4o-mini",  # Cheapest model: $0.15/$0.60 per 1M tokens (input/output)
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,  # Low temperature for deterministic parsing
            max_tokens=200  # Limit output to minimize costs
        )
        
        result_text = response.choices[0].message.content.strip()
        # Remove markdown code blocks if present
        result_text = re.sub(r'^```json\s*', '', result_text)
        result_text = re.sub(r'^```\s*', '', result_text)
        result_text = re.sub(r'```\s*$', '', result_text)
        
        parsed = json.loads(result_text)
        
        # Handle array response (chained instructions from LLM)
        if isinstance(parsed, list):
            # Mark all as LLM-used and return first with chain info
            if parsed:
                for p in parsed:
                    p["_llm_used"] = True
                parsed[0]["_is_chained"] = True
                parsed[0]["_chain"] = parsed
                return parsed[0]
            else:
                return {"op": "unknown", "raw": instr}
        
        parsed["_llm_used"] = True  # Mark that LLM was used
        return parsed
        
    except Exception as e:
        # Silently fail - fall back to rule-based
        return None


def is_simple_instruction(instr: str) -> bool:
    """
    Check if instruction is simple enough for rule-based parsing.
    Simple cases: very short, single operation words, clear patterns.
    """
    instr_lower = instr.lower().strip()
    
    # Very short instructions (likely simple)
    if len(instr) < 20:
        # Check for very simple patterns
        simple_patterns = [
            r'^shift\s+dates?\s+[+-]?\d+',  # "shift dates +7"
            r'^set\s+\w+\s+to\s+.+',  # "set X to Y"
            r'^change\s+\w+\s+to\s+.+',  # "change X to Y"
            r'^randomize\s+\w+\s+\d+-\d+',  # "randomize X 5-10"
        ]
        for pattern in simple_patterns:
            if re.match(pattern, instr_lower):
                return True
    
    return False


def parse_chained_instructions(instr: str, available_titles: List[str] = None, use_llm: bool = True) -> List[Dict[str, Any]]:
    """
    Parse chained instructions (multiple edits separated by 'and', ',', ';', etc.)
    Returns list of parsed instructions.
    """
    # Split by common chain separators
    # Pattern: split on " and ", ", ", "; ", " then ", " also "
    # But be careful with quoted strings and dates
    separators = [
        r'\s+and\s+',
        r',\s+',
        r';\s+',
        r'\s+then\s+',
        r'\s+also\s+',
    ]
    
    # Try to split intelligently
    parts = [instr]
    for sep in separators:
        new_parts = []
        for part in parts:
            # Don't split inside quotes
            splits = re.split(sep, part, flags=re.IGNORECASE)
            new_parts.extend([s.strip() for s in splits if s.strip()])
        parts = new_parts
    
    # Parse each instruction
    parsed_instructions = []
    for part in parts:
        if part:
            parsed = parse_instruction(part, available_titles, use_llm)
            if parsed.get("op") not in ("unknown", "none"):
                parsed_instructions.append(parsed)
    
    return parsed_instructions if parsed_instructions else [{"op": "unknown", "raw": instr}]


def parse_instruction(instr: str, available_titles: List[str] = None, use_llm: bool = True) -> Dict[str, Any]:
    """
    Parse user instruction. Uses LLM as primary parser, falls back to rule-based for simple cases.
    
    Args:
        instr: User instruction string
        available_titles: List of available field/column titles (for LLM context)
        use_llm: If True, use LLM as primary parser (default: True)
    """
    instr = instr.strip()
    if not instr:
        return {"op": "none"}

    # Check for domain-specific operations first (cannabinoid formulas, conversions, tiers)
    formula_op = parse_formula_edit(instr)
    if formula_op:
        return formula_op
    
    conversion_op = parse_conversion_edit(instr)
    if conversion_op:
        return conversion_op
    
    tier_op = parse_tier_adjustment(instr)
    if tier_op:
        return tier_op

    # Check if instruction is simple enough for rule-based (fast path for obvious cases)
    if is_simple_instruction(instr):
        # Try rule-based parsing for simple cases
        shift = parse_shift_days(instr)
        if shift is not None:
            return {"op": "shift_dates", "days": shift, "_method": "rule-based"}

        rnd = parse_randomize(instr)
        if rnd:
            rnd["_method"] = "rule-based"
            return rnd

        stv = parse_set(instr)
        if stv:
            stv["_method"] = "rule-based"
            return stv

    # Use LLM as primary parser (for complex/natural language instructions)
    if use_llm:
        llm_result = parse_instruction_with_llm(instr, available_titles)
        if llm_result and llm_result.get("op") != "unknown":
            return llm_result

    # If LLM failed or not enabled, try rule-based as fallback
    shift = parse_shift_days(instr)
    if shift is not None:
        return {"op": "shift_dates", "days": shift, "_method": "rule-based"}

    rnd = parse_randomize(instr)
    if rnd:
        rnd["_method"] = "rule-based"
        return rnd

    stv = parse_set(instr)
    if stv:
        stv["_method"] = "rule-based"
        return stv

    return {"op": "unknown", "raw": instr, "_method": "none"}


# -----------------------------
# Fuzzy match
# -----------------------------

def fuzzy_best_matches(query: str, choices: List[str], limit: int = 5) -> List[Tuple[str, float]]:
    qn = normalize_title(query)
    if not qn:
        return []
    if not choices:
        return []

    if HAS_RAPIDFUZZ:
        res = rf_process.extract(
            qn,
            choices,
            scorer=rf_fuzz.WRatio,
            limit=limit,
        )
        return [(r[0], float(r[1])) for r in res]
    else:
        # difflib fallback
        scored = []
        for c in choices:
            ratio = difflib.SequenceMatcher(None, qn, c).ratio() * 100.0
            scored.append((c, ratio))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:limit]


def fuzzy_match_text(query: str, text: str, threshold: float = 60.0) -> bool:
    """
    Check if query fuzzy matches text content.
    Returns True if similarity is above threshold.
    """
    query_norm = normalize_title(query)
    text_norm = normalize_title(text)
    
    if not query_norm or not text_norm:
        return query.lower() in text.lower()
    
    if HAS_RAPIDFUZZ:
        score = rf_fuzz.WRatio(query_norm, text_norm)
        return score >= threshold
    else:
        ratio = difflib.SequenceMatcher(None, query_norm, text_norm).ratio() * 100.0
        return ratio >= threshold


def resolve_text_directly(doc: fitz.Document, search_text: str, threshold: float = 60.0) -> List[Target]:
    """
    Search ALL text spans in the PDF for a given string, not just indexed titles.
    Uses fuzzy matching to find similar text.
    Returns list of targets that can be edited.
    """
    targets = []
    search_lower = search_text.lower().strip()
    
    for pno in range(len(doc)):
        page = doc[pno]
        spans = extract_spans(page)
        
        for sp in spans:
            text = sp["text"].strip()
            if not text or len(text) < 2:
                continue
            
            # Check for exact match first
            if search_lower in text.lower():
                style = pick_style_near_rect(spans, sp["bbox"])
                targets.append(
                    Target(
                        page_index=pno,
                        rect=sp["bbox"],
                        style=style,
                        original_text=text,
                        kind="direct_text"
                    )
                )
            # Then try fuzzy match
            elif fuzzy_match_text(search_text, text, threshold):
                style = pick_style_near_rect(spans, sp["bbox"])
                targets.append(
                    Target(
                        page_index=pno,
                        rect=sp["bbox"],
                        style=style,
                        original_text=text,
                        kind="direct_text"
                    )
                )
    
    return targets


# -----------------------------
# Transform utilities
# -----------------------------

NUM_RE = re.compile(r"^-?\d+(\.\d+)?$")

def format_number(value: float, decimals: int, percent: bool) -> str:
    fmt = f"{{:.{decimals}f}}"
    s = fmt.format(value)
    if percent and not s.endswith("%"):
        s = s + "%"
    return s


def stable_rng(seed_key: str) -> random.Random:
    # deterministic seed from string
    h = 0
    for ch in seed_key:
        h = (h * 131 + ord(ch)) & 0xFFFFFFFF
    return random.Random(h)


def extract_date_with_context(s: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Extract date portion from text that may contain labels.
    Returns (prefix, date_text, suffix) preserving original spacing.
    Returns (None, None, None) if no date found.
    """
    s_orig = s  # Keep original for spacing
    s = s.strip()
    # Try MM/DD/YYYY pattern
    m = DATE_PATTERNS[0].search(s)
    if m:
        date_text = m.group(0)
        prefix_raw = s[:m.start()]
        suffix_raw = s[m.end():]
        # Preserve trailing space from prefix if it exists
        prefix = prefix_raw.rstrip() if prefix_raw.strip() else None
        suffix = suffix_raw.lstrip() if suffix_raw.strip() else None
        # Add back spacing: if prefix had trailing space, preserve it
        if prefix and prefix_raw.endswith(" "):
            prefix = prefix + " "
        elif prefix and prefix_raw.endswith("  "):  # double space
            prefix = prefix + "  "
        return (prefix, date_text, suffix)
    # Try YYYY-MM-DD pattern
    m = DATE_PATTERNS[1].search(s)
    if m:
        date_text = m.group(0)
        prefix_raw = s[:m.start()]
        suffix_raw = s[m.end():]
        prefix = prefix_raw.rstrip() if prefix_raw.strip() else None
        suffix = suffix_raw.lstrip() if suffix_raw.strip() else None
        if prefix and prefix_raw.endswith(" "):
            prefix = prefix + " "
        elif prefix and prefix_raw.endswith("  "):
            prefix = prefix + "  "
        return (prefix, date_text, suffix)
    return (None, None, None)


def try_parse_date(s: str) -> Optional[datetime]:
    s = s.strip()
    # MM/DD/YYYY
    m = DATE_PATTERNS[0].search(s)
    if m:
        mm = int(m.group(1))
        dd = int(m.group(2))
        yy = int(m.group(3))
        if yy < 100:
            yy += 2000
        try:
            return datetime(yy, mm, dd)
        except Exception:
            return None
    # YYYY-MM-DD
    m = DATE_PATTERNS[1].search(s)
    if m:
        yy = int(m.group(1))
        mm = int(m.group(2))
        dd = int(m.group(3))
        try:
            return datetime(yy, mm, dd)
        except Exception:
            return None
    return None


def format_like_original_date(original: str, dt: datetime) -> str:
    original = original.strip()
    if DATE_PATTERNS[1].search(original):
        return f"{dt.year:04d}-{dt.month:02d}-{dt.day:02d}"
    # default to M/D/YYYY if original had single-digit possibilities; else MM/DD/YYYY
    m = DATE_PATTERNS[0].search(original)
    if m:
        mm_raw = m.group(1)
        dd_raw = m.group(2)
        yyyy = dt.year
        if len(mm_raw) == 1:
            mm = str(dt.month)
        else:
            mm = f"{dt.month:02d}"
        if len(dd_raw) == 1:
            dd = str(dt.day)
        else:
            dd = f"{dt.day:02d}"
        return f"{mm}/{dd}/{yyyy:04d}"
    return f"{dt.month:02d}/{dt.day:02d}/{dt.year:04d}"


# -----------------------------
# Apply edits in-place (redact + insert)
# -----------------------------

def apply_targets_in_place(
    doc: fitz.Document,
    targets: List[Target],
    new_text_for_target: List[str],
) -> None:
    """
    Batch per-page:
    - add redactions for all target rects
    - apply redactions once
    - insert replacement text
    """
    by_page: Dict[int, List[Tuple[Target, str]]] = {}
    for t, nt in zip(targets, new_text_for_target):
        by_page.setdefault(t.page_index, []).append((t, nt))

    for pno, items in by_page.items():
        page = doc[pno]
        # add redactions
        for (t, nt) in items:
            page.add_redact_annot(t.rect, fill=REDACT_FILL_RGB)
        page.apply_redactions(images=fitz.PDF_REDACT_IMAGE_NONE)

        # insert replacements
        spans = extract_spans(page)  # updated spans after redaction is fine; we already stored style
        for (t, nt) in items:
            style = t.style
            # If style font isn't available, PyMuPDF will fall back; that's ok.
            insert_text_fit(page, t.rect, nt, style, align=fitz.TEXT_ALIGN_LEFT)


# -----------------------------
# Resolve instruction -> targets
# -----------------------------

def resolve_title_to_entry(index: List[TitleEntry], user_title: str, doc: fitz.Document = None) -> Tuple[Optional[TitleEntry], Dict[str, Any]]:
    """
    Fuzzy match user title to index entry normalized keys.
    Falls back to direct text search if indexed search fails and doc is provided.
    Returns entry + debug info. If ambiguous/low score, returns None.
    """
    # Build choice list of entry.normalized
    choices = [e.normalized for e in index]
    matches = fuzzy_best_matches(user_title, choices, limit=5)
    debug = {"query": user_title, "query_norm": normalize_title(user_title), "matches": matches, "method": "indexed"}

    if not matches:
        debug["reason"] = "no_matches"
        # Fall back to direct text search if doc provided
        if doc:
            direct_targets = resolve_text_directly(doc, user_title, threshold=50.0)  # Lower threshold for direct search
            if direct_targets:
                debug["method"] = "direct_search"
                debug["targets_found"] = len(direct_targets)
                debug["reason"] = "found_via_direct_search"
                # Create a temporary entry for direct search results
                entry = TitleEntry(
                    title=user_title,
                    normalized=normalize_title(user_title),
                    targets=direct_targets,
                    meta={"type": "direct_search", "aliases": [user_title], "search_method": "direct"}
                )
                return entry, debug
        return None, debug

    best_key, best_score = matches[0]
    if best_score < DEFAULT_FUZZY_THRESHOLD:
        debug["reason"] = "below_threshold"
        # Fall back to direct text search if doc provided
        if doc:
            direct_targets = resolve_text_directly(doc, user_title, threshold=50.0)
            if direct_targets:
                debug["method"] = "direct_search"
                debug["targets_found"] = len(direct_targets)
                debug["reason"] = "found_via_direct_search"
                entry = TitleEntry(
                    title=user_title,
                    normalized=normalize_title(user_title),
                    targets=direct_targets,
                    meta={"type": "direct_search", "aliases": [user_title], "search_method": "direct"}
                )
                return entry, debug
        return None, debug

    # ambiguity check
    if len(matches) >= 2 and (matches[0][1] - matches[1][1]) <= AMBIGUITY_DELTA:
        debug["reason"] = "ambiguous"
        return None, debug

    # return entry by normalized key
    for e in index:
        if e.normalized == best_key:
            return e, debug

    debug["reason"] = "not_found_after_match"
    return None, debug


def generate_random_replacements_for_targets(
    targets: List[Target],
    min_v: float,
    max_v: float,
    decimals: int,
    percent: bool,
    seed_context: str,
) -> List[str]:
    out = []
    for i, t in enumerate(targets):
        rng = stable_rng(f"{seed_context}|p{t.page_index}|{t.rect.x0:.2f},{t.rect.y0:.2f}|{i}")
        v = rng.uniform(min_v, max_v)
        s = format_number(v, decimals, percent)
        out.append(s)
    return out


def generate_set_replacements_for_targets(targets: List[Target], value: str) -> List[str]:
    """
    Generate replacement values. For date targets, preserve prefix/suffix labels.
    """
    out = []
    for t in targets:
        if t.kind == "date_span":
            # For dates, try to preserve any prefix/suffix (like "Collected: " or labels)
            prefix, date_text, suffix = extract_date_with_context(t.original_text)
            if prefix is not None or suffix is not None:
                # Reconstruct with preserved prefix/suffix and spacing
                result = ""
                if prefix:
                    result += prefix
                    # If prefix doesn't end with space, add one (unless value starts with space)
                    if not prefix.endswith(" ") and not value.startswith(" "):
                        result += " "
                result += value
                if suffix:
                    # If value doesn't end with space, add one before suffix
                    if not value.endswith(" ") and not suffix.startswith(" "):
                        result += " "
                    result += suffix
                out.append(result)
            else:
                # No prefix/suffix, just use the value
                out.append(value)
        else:
            # For non-date targets, just use the value
            out.append(value)
    return out


def apply_intelligent_cannabinoid_edit(
    index: List[TitleEntry],
    cannabinoid_name: str,
    new_percent_value: str,
    doc: fitz.Document
) -> List[Tuple[List[Target], List[str], str]]:
    """
    Intelligently edit a cannabinoid value with automatic:
    1. % ↔ mg/g conversion (1% = 10 mg/g)
    2. Formula recalculation (Total THC = THCa × 0.877 + Δ9)
    3. Total recalculation (Total Cannabinoids = sum of components)
    
    Returns list of (targets, new_values, description) tuples for all related edits.
    """
    results = []
    
    # Extract numeric value from input
    percent_val = extract_numeric_value(new_percent_value)
    if percent_val is None:
        return results  # Can't proceed without numeric value
    
    # Normalize cannabinoid name
    cannabinoid_name_norm = normalize_cannabinoid_name(cannabinoid_name)
    if not cannabinoid_name_norm:
        cannabinoid_name_norm = cannabinoid_name
    
    # Find the cannabinoid entry
    cannabinoid_entry = None
    for entry in index:
        if entry.title == cannabinoid_name_norm or normalize_title(entry.title) == normalize_title(cannabinoid_name):
            cannabinoid_entry = entry
            break
    
    if not cannabinoid_entry or not cannabinoid_entry.targets:
        return results
    
    # Step 1: Update the % value(s) in the cannabinoid entry
    percent_targets = []
    percent_new_vals = []
    mg_per_g_targets = []
    mg_per_g_new_vals = []
    
    for target in cannabinoid_entry.targets:
        original = target.original_text.strip()
        # Check if this target contains a % value
        if '%' in original or re.search(r'\d+\.?\d*\s*%', original):
            # Format the new value preserving original style
            formatted_val = format_numeric_value(percent_val, original, decimals=2)
            percent_targets.append(target)
            percent_new_vals.append(formatted_val)
        # Also check for mg/g values that need conversion
        elif 'mg/g' in original.lower() or 'mg per g' in original.lower() or re.search(r'\d+\.?\d*\s*mg', original.lower()):
            # Convert % to mg/g: 1% = 10 mg/g
            mg_per_g_val = percent_val * 10.0
            formatted_mg = format_numeric_value(mg_per_g_val, original, decimals=2)
            mg_per_g_targets.append(target)
            mg_per_g_new_vals.append(formatted_mg)
    
    if percent_targets:
        results.append((percent_targets, percent_new_vals, f"Updated {cannabinoid_name_norm} to {new_percent_value}"))
    
    if mg_per_g_targets:
        results.append((mg_per_g_targets, mg_per_g_new_vals, f"Updated {cannabinoid_name_norm} mg/g to {percent_val * 10.0:.2f} mg/g"))
    
    # Step 2: Find and update related mg/g values in separate entries
    # Look for mg/g entries with same cannabinoid name (e.g., "THCa mg/g")
    cannabinoid_base = cannabinoid_name_norm.replace(' %', '').replace('%', '').strip()
    for entry in index:
        title_lower = entry.title.lower()
        cannabinoid_base_lower = cannabinoid_base.lower()
        # Check if this entry is for the same cannabinoid but in mg/g format
        if ('mg/g' in title_lower or 'mg per g' in title_lower) and cannabinoid_base_lower in title_lower:
            if entry.targets:
                mg_per_g_val = percent_val * 10.0  # 1% = 10 mg/g
                mg_targets = []
                mg_new_vals = []
                for target in entry.targets:
                    formatted_mg = format_numeric_value(mg_per_g_val, target.original_text, decimals=2)
                    mg_targets.append(target)
                    mg_new_vals.append(formatted_mg)
                if mg_targets:
                    results.append((mg_targets, mg_new_vals, f"Updated {entry.title} to {mg_per_g_val:.2f} mg/g"))
            break
    
    # Step 3: Recalculate Total THC if THCa or Δ9 was changed
    if cannabinoid_name_norm in ["THCa %", "Δ9 %"]:
        thca_entry, _ = resolve_title_to_entry(index, "THCa %", doc)
        delta9_entry, _ = resolve_title_to_entry(index, "Δ9 %", doc)
        total_thc_entry, _ = resolve_title_to_entry(index, "Total THC %", doc)
        
        if thca_entry and delta9_entry and total_thc_entry:
            # Get current values
            thca_val = None
            delta9_val = None
            
            if cannabinoid_name_norm == "THCa %":
                thca_val = percent_val
                # Get Δ9 from its entry
                if delta9_entry.targets:
                    delta9_val = extract_numeric_value(delta9_entry.targets[0].original_text)
            else:  # Δ9 %
                delta9_val = percent_val
                # Get THCa from its entry
                if thca_entry.targets:
                    thca_val = extract_numeric_value(thca_entry.targets[0].original_text)
            
            if thca_val is not None and delta9_val is not None:
                # Calculate: Total THC = THCa × 0.877 + Δ9
                total_thc_val = (thca_val * 0.877) + delta9_val
                
                # Update Total THC targets
                if total_thc_entry.targets:
                    total_thc_targets = []
                    total_thc_new_vals = []
                    for target in total_thc_entry.targets:
                        formatted_total = format_numeric_value(total_thc_val, target.original_text, decimals=2)
                        total_thc_targets.append(target)
                        total_thc_new_vals.append(formatted_total)
                    if total_thc_targets:
                        results.append((total_thc_targets, total_thc_new_vals, f"Recalculated Total THC: {total_thc_val:.2f}%"))
    
    # Step 4: Recalculate Total Cannabinoids (sum of all cannabinoids)
    # Find all cannabinoid entries and sum them
    total_cannabinoids_entry, _ = resolve_title_to_entry(index, "Total Cannabinoids %", doc)
    if total_cannabinoids_entry:
        cannabinoid_sum = 0.0
        cannabinoid_entries_found = []
        
        # Collect values from major cannabinoids
        for cb_name in ["THCa %", "Δ9 %", "CBD %", "CBN %", "CBG %"]:
            cb_entry, _ = resolve_title_to_entry(index, cb_name, doc)
            if cb_entry and cb_entry.targets:
                # Use updated value if this is the one we just changed
                if cb_name == cannabinoid_name_norm:
                    cb_val = percent_val
                else:
                    cb_val = extract_numeric_value(cb_entry.targets[0].original_text)
                if cb_val is not None:
                    cannabinoid_sum += cb_val
                    cannabinoid_entries_found.append(cb_name)
        
        if cannabinoid_sum > 0 and total_cannabinoids_entry.targets:
            total_cannabinoids_targets = []
            total_cannabinoids_new_vals = []
            for target in total_cannabinoids_entry.targets:
                formatted_total = format_numeric_value(cannabinoid_sum, target.original_text, decimals=2)
                total_cannabinoids_targets.append(target)
                total_cannabinoids_new_vals.append(formatted_total)
            if total_cannabinoids_targets:
                results.append((total_cannabinoids_targets, total_cannabinoids_new_vals, 
                              f"Recalculated Total Cannabinoids: {cannabinoid_sum:.2f}%"))
    
    return results


def shift_dates_for_targets(targets: List[Target], days: int) -> List[str]:
    out = []
    for t in targets:
        dt = try_parse_date(t.original_text)
        if dt is None:
            # if not parseable, leave as-is
            out.append(t.original_text)
            continue
        nd = dt + timedelta(days=days)
        out.append(format_like_original_date(t.original_text, nd))
    return out


def extract_numeric_value(text: str) -> Optional[float]:
    """Extract numeric value from text, handling percentages and units."""
    # Remove common text, keep numbers and decimal points
    cleaned = re.sub(r'[^\d.]', '', text.replace('%', '').strip())
    try:
        return float(cleaned) if cleaned else None
    except ValueError:
        return None


def format_numeric_value(value: float, original_text: str, decimals: int = 2) -> str:
    """Format numeric value preserving original format (percentage, units, etc.)."""
    formatted = f"{value:.{decimals}f}"
    
    # Preserve percentage sign if original had it
    if '%' in original_text:
        return f"{formatted}%"
    
    # Preserve mg/g if original had it
    if 'mg/g' in original_text.lower() or 'mg / g' in original_text.lower():
        return f"{formatted} mg/g"
    
    return formatted


def calculate_thc_formula(thca_entry: TitleEntry, delta9_entry: TitleEntry, total_thc_entry: TitleEntry) -> Tuple[List[Target], List[str]]:
    """
    Calculate Total THC = THCa × 0.877 + Δ9 for all matching pairs.
    Returns targets and new values for Total THC.
    """
    # Match THCa and Δ9 values by position (assuming they're in same table rows)
    # Strategy: Group by page and approximate y-position to match rows
    
    thca_targets = thca_entry.targets
    delta9_targets = delta9_entry.targets
    total_thc_targets = total_thc_entry.targets
    
    # Group targets by page and approximate row (y-position)
    def get_row_key(target: Target) -> Tuple[int, int]:
        return (target.page_index, int(target.rect.y0 / 20))  # Group by ~20pt rows
    
    thca_by_row = {}
    for t in thca_targets:
        key = get_row_key(t)
        if key not in thca_by_row:
            thca_by_row[key] = []
        thca_by_row[key].append(t)
    
    delta9_by_row = {}
    for t in delta9_targets:
        key = get_row_key(t)
        if key not in delta9_by_row:
            delta9_by_row[key] = []
        delta9_by_row[key].append(t)
    
    total_thc_by_row = {}
    for t in total_thc_targets:
        key = get_row_key(t)
        if key not in total_thc_by_row:
            total_thc_by_row[key] = []
        total_thc_by_row[key].append(t)
    
    # Calculate Total THC for each row
    result_targets = []
    result_values = []
    
    for row_key in sorted(set(list(thca_by_row.keys()) + list(delta9_by_row.keys()))):
        thca_vals = [extract_numeric_value(t.original_text) for t in thca_by_row.get(row_key, [])]
        delta9_vals = [extract_numeric_value(t.original_text) for t in delta9_by_row.get(row_key, [])]
        
        # Use first available values
        thca_val = next((v for v in thca_vals if v is not None), None)
        delta9_val = next((v for v in delta9_vals if v is not None), None)
        
        if thca_val is not None and delta9_val is not None:
            # Calculate: Total THC = THCa × 0.877 + Δ9
            total_thc_val = (thca_val * 0.877) + delta9_val
            
            # Find corresponding Total THC targets
            for target in total_thc_by_row.get(row_key, []):
                result_targets.append(target)
                result_values.append(format_numeric_value(total_thc_val, target.original_text, decimals=2))
    
    return result_targets, result_values


def reverse_engineer_totals(index: List[TitleEntry]) -> List[Tuple[List[Target], List[str]]]:
    """
    Reverse-engineer component values so totals add up correctly.
    Finds entries that look like totals and adjusts components.
    """
    # Find entries that might be totals (contain "Total", "Sum", etc.)
    total_entries = []
    component_entries = []
    
    for entry in index:
        title_lower = entry.title.lower()
        if 'total' in title_lower or 'sum' in title_lower:
            total_entries.append(entry)
        elif entry.meta.get("type") in ("field", "table_column", "cannabinoid"):
            # Exclude totals from components
            if 'total' not in title_lower and 'sum' not in title_lower:
                component_entries.append(entry)
    
    results = []
    
    # For each total, try to find components that should sum to it
    # This is heuristic - we'll look for entries on the same page/table
    for total_entry in total_entries:
        if not total_entry.targets:
            continue
        
        # Group by page
        for page_idx in set(t.page_index for t in total_entry.targets):
            page_total_targets = [t for t in total_entry.targets if t.page_index == page_idx]
            
            for total_target in page_total_targets:
                total_val = extract_numeric_value(total_target.original_text)
                if total_val is None or total_val <= 0:
                    continue
                
                # Find nearby component targets (same page, similar y-position)
                # Look for components in same table/region
                nearby_components = []
                for comp_entry in component_entries:
                    if comp_entry == total_entry:
                        continue
                    # Check if entry name suggests it's a component of this total
                    comp_title_lower = comp_entry.title.lower()
                    total_title_lower = total_entry.title.lower()
                    
                    # Match if component name appears in total name (e.g., "Total Cannabinoids" contains "Cannabinoids")
                    is_related = False
                    if 'total cannabinoids' in total_title_lower:
                        # Look for individual cannabinoid entries
                        if any(cb in comp_title_lower for cb in ['thca', 'thc', 'cbd', 'cbn', 'cbg']):
                            is_related = True
                    elif 'total thc' in total_title_lower:
                        # Look for THCa and Δ9
                        if any(cb in comp_title_lower for cb in ['thca', 'delta', 'δ9', 'd9']):
                            is_related = True
                    else:
                        # Generic: check if component name is in total name
                        words = comp_title_lower.split()
                        if any(word in total_title_lower for word in words if len(word) > 3):
                            is_related = True
                    
                    for comp_target in comp_entry.targets:
                        if (comp_target.page_index == page_idx and 
                            abs(comp_target.rect.y0 - total_target.rect.y0) < 100):  # Within 100pt
                            comp_val = extract_numeric_value(comp_target.original_text)
                            if comp_val is not None and comp_val > 0:
                                # Prefer related components, but include nearby ones too
                                priority = 1 if is_related else 2
                                nearby_components.append((priority, comp_entry, comp_target, comp_val))
                
                if len(nearby_components) >= 2:
                    # Sort by priority (related first) and proximity
                    nearby_components.sort(key=lambda x: (x[0], abs(x[2].rect.y0 - total_target.rect.y0)))
                    # Take top components (up to 10)
                    selected_components = nearby_components[:10]
                    
                    # Adjust components proportionally to sum to total
                    current_sum = sum(comp_val for _, _, _, comp_val in selected_components)
                    if current_sum > 0 and abs(current_sum - total_val) > 0.01:  # Significant difference
                        ratio = total_val / current_sum
                        
                        # Create adjusted values
                        adjusted_targets = []
                        adjusted_values = []
                        for _, comp_entry, comp_target, comp_val in selected_components:
                            adjusted_val = comp_val * ratio
                            adjusted_targets.append(comp_target)
                            adjusted_values.append(format_numeric_value(adjusted_val, comp_target.original_text, decimals=2))
                        
                        results.append((adjusted_targets, adjusted_values))
    
    return results


def fix_percent_mg_conversions(index: List[TitleEntry]) -> List[Tuple[List[Target], List[str]]]:
    """
    Fix % ↔ mg/g conversions. Standard conversion: 1% = 10 mg/g
    """
    results = []
    
    # Find entries with % and mg/g
    percent_entries = []
    mgg_entries = []
    
    for entry in index:
        title_lower = entry.title.lower()
        if '%' in entry.title or any('%' in t.original_text for t in entry.targets):
            percent_entries.append(entry)
        if 'mg/g' in title_lower or 'mg / g' in title_lower or any('mg/g' in t.original_text.lower() for t in entry.targets):
            mgg_entries.append(entry)
    
    # Fix conversions: % to mg/g (multiply by 10) or mg/g to % (divide by 10)
    for entry in percent_entries:
        for target in entry.targets:
            val = extract_numeric_value(target.original_text)
            if val is not None and '%' in target.original_text:
                # Check if there's a corresponding mg/g value that doesn't match
                # For now, we'll just ensure consistency - if user wants to fix, they can specify
                pass
    
    # More sophisticated: match % and mg/g entries by name similarity
    for percent_entry in percent_entries:
        percent_name_base = re.sub(r'[%\s]', '', percent_entry.title.lower())
        
        for mgg_entry in mgg_entries:
            mgg_name_base = re.sub(r'[mg/g\s]', '', mgg_entry.title.lower())
            
            # If names are similar, check if conversion is correct
            if percent_name_base == mgg_name_base or (len(percent_name_base) > 3 and percent_name_base in mgg_name_base):
                # Match targets by position
                for p_target in percent_entry.targets:
                    p_val = extract_numeric_value(p_target.original_text)
                    if p_val is None:
                        continue
                    
                    for m_target in mgg_entry.targets:
                        if (m_target.page_index == p_target.page_index and 
                            abs(m_target.rect.y0 - p_target.rect.y0) < 30):
                            m_val = extract_numeric_value(m_target.original_text)
                            if m_val is not None:
                                # Check conversion: 1% = 10 mg/g
                                expected_mg = p_val * 10
                                if abs(m_val - expected_mg) > 0.1:  # Significant difference
                                    results.append(([m_target], [format_numeric_value(expected_mg, m_target.original_text, decimals=2)]))
    
    return results


def extract_strain_and_date_from_doc(doc: fitz.Document) -> Tuple[str, str]:
    """
    Extract strain name and date from PDF document for filename generation.
    Reads directly from document to get current values (after edits).
    Returns (strain_name, date) as strings suitable for filename.
    """
    strain_name = "Unknown"
    date_str = "Unknown"
    
    # Extract all text from first page (where COA info usually is)
    for pno in range(min(2, len(doc))):  # Check first 2 pages
        page = doc[pno]
        text = page.get_text()
        lines = text.split('\n')
        
        # Look for strain name
        strain_patterns = [
            r'strain[:\s]+([^\n:]+?)(?:\s*$|\s*[:\n])',  # Stop at newline, colon, or end
            r'strain\s+name[:\s]+([^\n:]+?)(?:\s*$|\s*[:\n])',
            r'variety[:\s]+([^\n:]+?)(?:\s*$|\s*[:\n])',
        ]
        for line in lines:
            line_lower = line.lower().strip()
            for pattern in strain_patterns:
                match = re.search(pattern, line_lower, re.IGNORECASE)
                if match:
                    strain_value = match.group(1).strip()
                    # Remove common suffixes that might be captured
                    strain_value = re.sub(r'\s*(\(.*?\)|,.*?$).*$', '', strain_value)
                    if strain_value and len(strain_value) > 1:
                        # Clean up for filename (remove special chars, spaces -> underscores)
                        strain_name = re.sub(r'[^\w\s-]', '', strain_value)
                        strain_name = re.sub(r'\s+', '_', strain_name.strip())
                        strain_name = strain_name[:50]  # Limit length
                        if strain_name and strain_name != "Unknown":
                            break
            if strain_name != "Unknown":
                break
        if strain_name != "Unknown":
            break
        
        # Look for date (prefer "Date Tested", "Collected", etc.)
        date_patterns = [
            r'date\s+tested[:\s]+([^\n:]+?)(?:\s*$|\s*[:\n])',
            r'collected[:\s]+([^\n:]+?)(?:\s*$|\s*[:\n])',
            r'test\s+date[:\s]+([^\n:]+?)(?:\s*$|\s*[:\n])',
            r'\bdate[:\s]+(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b',
        ]
        for line in lines:
            line_lower = line.lower().strip()
            for pattern in date_patterns:
                match = re.search(pattern, line_lower, re.IGNORECASE)
                if match:
                    date_value = match.group(1).strip()
                    # Remove common prefixes/suffixes
                    date_value = re.sub(r'^(collected|received|tested|completed)[:\s]*', '', date_value, flags=re.IGNORECASE)
                    date_value = date_value.strip()
                    if date_value:
                        # Try to parse date
                        dt = try_parse_date(date_value)
                        if dt:
                            # Format as YYYY-MM-DD for filename
                            date_str = dt.strftime("%Y-%m-%d")
                            break
                        else:
                            # Use raw value if can't parse, clean it up
                            date_str = re.sub(r'[^\w\s/-]', '', date_value)
                            date_str = re.sub(r'\s+', '_', date_str.strip())
                            date_str = date_str[:20]  # Limit length
                            if date_str:
                                break
            if date_str != "Unknown":
                break
        if date_str != "Unknown":
            break
    
    # If still no date found, use current date as fallback
    if date_str == "Unknown":
        date_str = datetime.now().strftime("%Y-%m-%d")
    
    return strain_name, date_str


def tier_adjust_values(index: List[TitleEntry], tier: str) -> List[Tuple[List[Target], List[str]]]:
    """
    Adjust values based on tier ranges (low/mid/high).
    Tier ranges (example for percentages):
    - Low: 10-15%
    - Mid: 15-25%
    - High: 25-35%
    """
    # Define tier ranges (can be customized)
    tier_ranges = {
        "low": {"min": 10.0, "max": 15.0},
        "mid": {"min": 15.0, "max": 25.0},
        "high": {"min": 25.0, "max": 35.0},
    }
    
    if tier not in tier_ranges:
        tier = "mid"
    
    range_min = tier_ranges[tier]["min"]
    range_max = tier_ranges[tier]["max"]
    
    results = []
    
    # Find numeric entries (especially cannabinoids)
    for entry in index:
        if entry.meta.get("type") == "cannabinoid" or '%' in entry.title:
            for target in entry.targets:
                val = extract_numeric_value(target.original_text)
                if val is not None:
                    # Adjust to tier range (randomize within range)
                    import random
                    new_val = random.uniform(range_min, range_max)
                    results.append(([target], [format_numeric_value(new_val, target.original_text, decimals=2)]))
    
    return results


# -----------------------------
# Streamlit UI
# -----------------------------

st.set_page_config(page_title="PDF Editor", layout="wide")
st.title("PDF Editor — In-place edits that preserve formatting")

st.write(
    "Upload a PDF, type an instruction like **'randomize LOD between 10-30%'** or **'set Sample ID to ABC'**, "
    "and download the edited PDF. Works best for PDFs with selectable text (not scanned images)."
)

uploaded = st.file_uploader("Upload PDF", type=["pdf"])

# Check if API key is available
api_key_available = False
try:
    if hasattr(st, 'secrets') and 'openai' in st.secrets and 'api_key' in st.secrets['openai']:
        api_key_available = True
except Exception:
    pass

if not api_key_available:
    import os
    if os.getenv("OPENAI_API_KEY"):
        api_key_available = True

# LLM is primary parser (enabled by default)
use_llm = st.checkbox(
    "Use AI for instruction parsing (recommended)", 
    value=True,
    help="Uses GPT-4o-mini (cheapest model: $0.15/$0.60 per 1M tokens) as primary parser. Falls back to rule-based for very simple instructions. API key configured in .streamlit/secrets.toml"
)

if not api_key_available:
    st.warning("⚠️ OpenAI API key not found. LLM parsing will be disabled. Please set it in `.streamlit/secrets.toml` or `OPENAI_API_KEY` environment variable.")
    use_llm = False

instruction = st.text_input(
    "Edit instruction", 
    placeholder='e.g. change THCa % to 25.5 and ensure THCa × 0.877 + Δ9 = Total THC'
)
st.caption("💡 Common queries: Change cannabinoid values, ensure formulas are correct, reverse-engineer totals, fix % ↔ mg/g conversions, adjust by tier (low/mid/high)")

colA, colB = st.columns([1, 1])

if uploaded is not None:
    pdf_bytes = uploaded.read()

    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception as e:
        st.error(f"Could not open PDF: {e}")
        st.stop()

    with st.spinner("Parsing PDF into editable datapoints..."):
        index, debug = build_document_index(doc)

    titles = sorted({e.title for e in index})
    norm_titles = sorted({e.normalized for e in index})
    
    # Always prioritize these 4 cannabinoid values at the top
    priority_cannabinoids = ["THCa %", "Δ9 %", "Total THC %", "Total Cannabinoids %"]
    
    # Build display list with priority cannabinoids first
    priority_titles = []
    priority_titles_with_targets = []
    other_cannabinoid_titles = []
    other_titles = []
    
    # First, collect all entries
    for entry in index:
        title = entry.title
        if title in priority_cannabinoids:
            if entry.targets:
                priority_titles_with_targets.append(title)
        elif any(cb in title.lower() for cb in ['thca', 'thc', 'cannabinoid', 'δ9', 'delta', 'cbd', 'cbn', 'cbg']):
            if entry.targets:
                other_cannabinoid_titles.append(title)
        else:
            if entry.targets:
                other_titles.append(title)
    
    # Always show the 4 priority cannabinoids at the top (in order)
    # Mark which ones have targets vs which can be edited via direct search
    for priority_name in priority_cannabinoids:
        priority_titles.append(priority_name)
    
    # Sort other groups
    other_cannabinoid_titles = sorted(set(other_cannabinoid_titles))
    other_titles = sorted(set(other_titles))
    
    # Build final display (priority always first, then others)
    display_titles = priority_titles + other_cannabinoid_titles + other_titles

    with colA:
        st.subheader("Detected editable titles & values")
        st.caption("Includes field labels, table headers, and detected cannabinoid values. Priority cannabinoids shown first.")
        st.info("💡 **You can edit ANY text in the PDF**, even if it's not listed here! Just type the text you want to change.")
        if display_titles:
            # Show priority cannabinoids at the top (always shown)
            if priority_titles:
                st.write("**🔝 Priority Cannabinoids (Most Edited):**")
                for pt in priority_titles:
                    # Check if it has targets
                    matching_entry = next((e for e in index if e.title == pt), None)
                    if matching_entry and matching_entry.targets:
                        count = len(matching_entry.targets)
                        st.write(f"  • **{pt}** ✓ ({count} value(s))")
                    else:
                        st.write(f"  • {pt} (can be edited via direct search)")
                st.write("")
            
            # Show other cannabinoids
            if other_cannabinoid_titles:
                st.write("**Other Cannabinoids:**")
                st.write(other_cannabinoid_titles)
                st.write("")
            
            # Show other fields
            if other_titles:
                st.write("**Other Fields:**")
                st.write(other_titles[:150] if len(other_titles) > 150 else other_titles)
                if len(other_titles) > 150:
                    st.caption(f"Showing first 150 of {len(other_titles)} other fields.")
        else:
            st.write("No titles detected (PDF may be scanned or text is not extractable).")
            st.info("💡 You can still edit text directly by typing what you want to change in the instruction field.")

    with colB:
        st.subheader("Debug (optional)")
        with st.expander("Show parsed index metadata"):
            st.code(json.dumps(debug, indent=2, default=str)[:20000])

    st.divider()

    if instruction:
        # Parse chained instructions (multiple edits)
        parsed_instructions = parse_chained_instructions(instruction, available_titles=titles, use_llm=use_llm)
        
        # Handle LLM returning array directly
        if parsed_instructions and parsed_instructions[0].get("_is_chained") and "_chain" in parsed_instructions[0]:
            parsed_instructions = parsed_instructions[0]["_chain"]
        
        st.subheader(f"Parsed {len(parsed_instructions)} instruction(s)")
        st.code(json.dumps(parsed_instructions, indent=2))
        
        # Check if any failed
        failed = [p for p in parsed_instructions if p.get("op") in ("unknown", "none")]
        if failed:
            st.error(f"Failed to parse {len(failed)} instruction(s). Try rephrasing.")
            doc.close()
            st.stop()
        
        # Track which titles will be modified (for visual feedback)
        modified_titles = set()
        
        # Process all instructions and collect edits
        all_edit_plans = []
        
        for parsed in parsed_instructions:
            if parsed.get("op") in ("unknown", "none"):
                continue
                
            plan_debug: Dict[str, Any] = {"parsed": parsed, "resolution": None, "targets_count": 0}
            
            if parsed["op"] == "shift_dates":
                entry, res_debug = resolve_title_to_entry(index, "dates", doc)
                plan_debug["resolution"] = res_debug
                if entry is None or not entry.targets:
                    continue
                
                targets = entry.targets
                new_vals = shift_dates_for_targets(targets, int(parsed["days"]))
                plan_debug["targets_count"] = len(targets)
                modified_titles.add("Dates")
                all_edit_plans.append({
                    "targets": targets,
                    "new_vals": new_vals,
                    "plan_debug": plan_debug,
                    "title": "Dates"
                })
                
            elif parsed["op"] == "formula_thc":
                # Calculate Total THC from THCa and Δ9: Total THC = THCa × 0.877 + Δ9
                thca_entry, _ = resolve_title_to_entry(index, "THCa %", doc)
                delta9_entry, _ = resolve_title_to_entry(index, "Δ9 %", doc)
                total_thc_entry, _ = resolve_title_to_entry(index, "Total THC %", doc)
                
                if thca_entry and delta9_entry and total_thc_entry:
                    targets, new_vals = calculate_thc_formula(thca_entry, delta9_entry, total_thc_entry)
                    if targets and new_vals:
                        plan_debug["targets_count"] = len(targets)
                        plan_debug["formula"] = "THCa × 0.877 + Δ9 = Total THC"
                        modified_titles.add("THCa %")
                        modified_titles.add("Δ9 %")
                        modified_titles.add("Total THC %")
                        all_edit_plans.append({
                            "targets": targets,
                            "new_vals": new_vals,
                            "plan_debug": plan_debug,
                            "title": "Total THC % (calculated)"
                        })
                    else:
                        st.warning("Could not calculate Total THC values. Make sure THCa and Δ9 values are present.")
                else:
                    missing = []
                    if not thca_entry:
                        missing.append("THCa %")
                    if not delta9_entry:
                        missing.append("Δ9 %")
                    if not total_thc_entry:
                        missing.append("Total THC %")
                    st.warning(f"Could not find required cannabinoid fields: {', '.join(missing)}")
                continue
                    
            elif parsed["op"] == "reverse_engineer_totals":
                # Reverse-engineer component values so totals add up correctly
                results = reverse_engineer_totals(index)
                if results:
                    for targets, new_vals in results:
                        plan_debug_copy = plan_debug.copy()
                        plan_debug_copy["targets_count"] = len(targets)
                        plan_debug_copy["operation"] = "reverse_engineer_totals"
                        # Mark all affected titles
                        for target in targets:
                            # Find entry for this target
                            for entry in index:
                                if target in entry.targets:
                                    modified_titles.add(entry.title)
                        all_edit_plans.append({
                            "targets": targets,
                            "new_vals": new_vals,
                            "plan_debug": plan_debug_copy,
                            "title": "Totals (reverse-engineered)"
                        })
                    st.success(f"Found {len(results)} total(s) to reverse-engineer.")
                else:
                    st.warning("Could not find totals to reverse-engineer. Make sure PDF contains 'Total' or 'Sum' fields.")
                continue
                
            elif parsed["op"] == "fix_conversions":
                # Fix % ↔ mg/g conversions
                results = fix_percent_mg_conversions(index)
                if results:
                    for targets, new_vals in results:
                        plan_debug_copy = plan_debug.copy()
                        plan_debug_copy["targets_count"] = len(targets)
                        plan_debug_copy["operation"] = "fix_conversions"
                        # Mark all affected titles
                        for target in targets:
                            for entry in index:
                                if target in entry.targets:
                                    modified_titles.add(entry.title)
                        all_edit_plans.append({
                            "targets": targets,
                            "new_vals": new_vals,
                            "plan_debug": plan_debug_copy,
                            "title": "Conversions (fixed)"
                        })
                    st.success(f"Fixed {len(results)} conversion(s).")
                else:
                    st.info("No conversion mismatches found. All % ↔ mg/g conversions appear correct.")
                continue
                
            elif parsed["op"] == "tier_adjustment":
                tier = parsed.get("tier", "mid")
                # Adjust values based on tier ranges
                results = tier_adjust_values(index, tier)
                if results:
                    for targets, new_vals in results:
                        plan_debug_copy = plan_debug.copy()
                        plan_debug_copy["targets_count"] = len(targets)
                        plan_debug_copy["tier"] = tier
                        # Mark all affected titles
                        for target in targets:
                            for entry in index:
                                if target in entry.targets:
                                    modified_titles.add(entry.title)
                        all_edit_plans.append({
                            "targets": targets,
                            "new_vals": new_vals,
                            "plan_debug": plan_debug_copy,
                            "title": f"Tier adjustment ({tier})"
                        })
                    st.success(f"Adjusted {len(results)} value(s) to {tier} tier range.")
                else:
                    st.warning(f"Could not find values to adjust for {tier} tier.")
                continue
                
            else:
                # set/randomize: resolve title
                user_title = parsed.get("title", "")
                entry, res_debug = resolve_title_to_entry(index, user_title, doc)
                plan_debug["resolution"] = res_debug
                
                # Show if direct search was used
                if res_debug.get("method") == "direct_search":
                    st.info(f"🔍 Found '{user_title}' via direct text search ({res_debug.get('targets_found', 0)} matches)")
                
                if entry is None:
                    continue
                
                if not entry.targets:
                    continue
                
                targets = entry.targets
                plan_debug["targets_count"] = len(targets)
                plan_debug["matched_title"] = entry.title
                plan_debug["matched_type"] = entry.meta.get("type")
                modified_titles.add(entry.title)
                
                # Generate replacements
                if parsed["op"] == "set":
                    value = parsed["value"]
                    
                    # Check if this is a cannabinoid edit that needs intelligent handling
                    entry_title_lower = entry.title.lower()
                    is_cannabinoid = any(cb in entry_title_lower for cb in ['thca', 'thc', 'cannabinoid', 'δ9', 'delta', 'cbd', 'cbn', 'cbg'])
                    is_percent_value = '%' in value or re.search(r'\d+\.?\d*\s*%', value, re.IGNORECASE)
                    
                    if is_cannabinoid and is_percent_value:
                        # Use intelligent cannabinoid edit handler
                        intelligent_edits = apply_intelligent_cannabinoid_edit(index, entry.title, value, doc)
                        
                        # Add the primary edit
                        new_vals = generate_set_replacements_for_targets(targets, value)
                        all_edit_plans.append({
                            "targets": targets,
                            "new_vals": new_vals,
                            "plan_debug": plan_debug,
                            "title": entry.title
                        })
                        
                        # Add all related edits (conversions, formulas, totals)
                        for intel_targets, intel_vals, intel_desc in intelligent_edits:
                            # Skip if these are the same targets we already added
                            if intel_targets != targets:
                                intel_plan_debug = {"description": intel_desc, "type": "intelligent_edit", "operation": "auto_calculated"}
                                all_edit_plans.append({
                                    "targets": intel_targets,
                                    "new_vals": intel_vals,
                                    "plan_debug": intel_plan_debug,
                                    "title": intel_desc
                                })
                                # Track modified titles for visual feedback
                                for intel_target in intel_targets:
                                    # Find which entry this target belongs to
                                    for e in index:
                                        if intel_target in e.targets:
                                            modified_titles.add(e.title)
                                            break
                    else:
                        # Regular set operation
                        new_vals = generate_set_replacements_for_targets(targets, value)
                        all_edit_plans.append({
                            "targets": targets,
                            "new_vals": new_vals,
                            "plan_debug": plan_debug,
                            "title": entry.title
                        })
                elif parsed["op"] == "randomize":
                    min_v = float(parsed["min"])
                    max_v = float(parsed["max"])
                    decimals = int(parsed["decimals"])
                    percent = bool(parsed["percent"])
                    
                    if (not percent) and ("% " in (entry.title + " ")) or entry.title.strip().endswith("%"):
                        percent = True
                    
                    if any(t.original_text.strip().endswith("%") for t in targets):
                        percent = True
                    
                    new_vals = generate_random_replacements_for_targets(
                        targets=targets,
                        min_v=min_v,
                        max_v=max_v,
                        decimals=decimals,
                        percent=percent,
                        seed_context=f"{uploaded.name}|{instruction}|{entry.normalized}",
                    )
                    all_edit_plans.append({
                        "targets": targets,
                        "new_vals": new_vals,
                        "plan_debug": plan_debug,
                        "title": entry.title
                    })
                else:
                    continue
        
        if not all_edit_plans:
            st.error("No valid edits could be applied. Check your instruction and try again.")
            doc.close()
            st.stop()
        
        # Show preview of all edits
        st.subheader("Edit Plan Preview")
        for i, plan in enumerate(all_edit_plans):
            st.write(f"**{i+1}. {plan['title']}**")
            preview_pairs = []
            for t, nv in list(zip(plan["targets"], plan["new_vals"]))[:5]:
                preview_pairs.append({"from": t.original_text, "to": nv, "page": t.page_index + 1})
            st.code(json.dumps(preview_pairs, indent=2, default=str))
        
        # Update titles display with green highlighting
        with colA:
            st.subheader("Detected editable titles")
            st.caption("These come from key/value labels and table headers inferred from the PDF.")
            
            # Display titles with green highlighting for modified ones
            titles_html = "<div style='max-height: 400px; overflow-y: auto;'>"
            for title in titles[:200]:
                if title in modified_titles:
                    titles_html += f"<div style='background-color: #90EE90; padding: 4px; margin: 2px; border-radius: 4px;'><strong>{title}</strong> ✓</div>"
                else:
                    titles_html += f"<div style='padding: 4px; margin: 2px;'>{title}</div>"
            titles_html += "</div>"
            st.markdown(titles_html, unsafe_allow_html=True)
            
            if len(titles) > 200:
                st.caption(f"Showing first 200 of {len(titles)} titles.")
        
        # Apply all edits
        if st.button("Apply all edits and generate PDF"):
            with st.spinner("Applying all edits in-place..."):
                # Combine all targets and values
                all_targets = []
                all_new_vals = []
                for plan in all_edit_plans:
                    all_targets.extend(plan["targets"])
                    all_new_vals.extend(plan["new_vals"])
                
                apply_targets_in_place(doc, all_targets, all_new_vals)
            
            # Extract strain and date from edited document (to get final values)
            strain_name, date_str = extract_strain_and_date_from_doc(doc)
            
            # Generate filename: COA_{strain-name}_{date}.pdf
            output_filename = f"COA_{strain_name}_{date_str}.pdf"
            
            out_bytes = doc.tobytes(garbage=4, deflate=True)
            st.success(f"Done! Applied {len(all_edit_plans)} edit(s) to {len(all_targets)} target(s). Download below.")
            st.info(f"📄 Filename: `{output_filename}`")
            
            st.download_button(
                label="Download edited PDF",
                data=out_bytes,
                file_name=output_filename,
                mime="application/pdf",
            )
        
        doc.close()
else:
    st.info("Upload a PDF to begin.")