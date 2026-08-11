"""Reconstruct a table's HTML from the PDF text layer (pdftext), CPU-only.

Ported from the datalab pdftext_backfill task, trimmed to the deterministic,
per-block core (no cross-page cluster consensus, no LLM orphan/header/garble
calls — those reduce to their documented deterministic fallbacks). Given the
pdftext spans inside a Table block's bbox, it sweeps several grid
parameterizations, scores them with a deterministic judge, and emits
``<table>`` HTML. Returns ``(html, score)`` or ``None`` when no grid resolves.

Input ``lines``: ``[(spans, y0, y1), ...]`` with ``spans = [(text, x0, x1), ...]``
in PDF points, matching the shape pdftext's dictionary_output yields per line.
"""

import itertools
import re
from collections import Counter, defaultdict
from html import escape
from statistics import median

MIN_CELLS_PER_ROW = 2  # a "data row" has >= this many tokens (2 supports two-column tables; the judge filters degenerate grids)
MIN_TABLE_ROWS = 3  # need at least this many data rows to treat it as a table
GAP_MIN_PT = 2.5  # projection: gaps narrower than this don't separate columns
PROJ_FRACS = (0.01, 0.03, 0.10)  # projection row-frequency thresholds to sweep
GRID_REGULARITY = 0.5

_PUA = re.compile("[\ue000-\uf8ff\ufffd]")  # private-use area + replacement char
_GARBLE_CHAR_FRAC = 0.02  # >2% private-use / replacement chars => garbled
# Leader runs (TOC dot leaders, form underscores/dashes) - not real cells;
# gridding them as columns explodes the column count, so they're dropped.
_LEADER_ONLY = re.compile(r"^[.·•…_\-\s]+$")
_NUMISH = re.compile(r"^-?[\d,.]*\d[\d,.]*%?$")
_NUM_TOK = re.compile(r"-?[\d,]+(\.\d+)?%?")
_YEAR = re.compile(r"^(19|20)\d\d$")
_PCTISH = re.compile(r"(less than|to less|%)")
_CODE_CELL = re.compile(r"^[A-Za-z0-9_./:\-\[\]{}()]+$")


# --------------------------------------------------------------------------- #
# garble gate (deterministic prefilter only)
# --------------------------------------------------------------------------- #
def _garble_ok(lines) -> bool:
    """True if the text layer looks clean. Deterministic: reject only on a high
    fraction of private-use / replacement glyphs (mojibake / bad encoding)."""
    text = " ".join(t for spans, _, _ in lines[:40] for t, _, _ in spans)
    if not text:
        return False
    return len(_PUA.findall(text)) / len(text) <= _GARBLE_CHAR_FRAC


# --------------------------------------------------------------------------- #
# header band split
# --------------------------------------------------------------------------- #
def _numfrac(spans) -> float:
    return sum(1 for t, _, _ in spans if _NUMISH.match(t)) / max(len(spans), 1)


def _find_header_band(lines):
    """y of the first DATA line (span-rich AND numeric-majority); everything
    above is the header band. None when the table has no numeric transition."""
    ys = [
        y0
        for spans, y0, _ in lines
        # Strictly more numeric than text: a 2-token (title, page#) TOC row is
        # exactly 0.5 and must not anchor the data region, or everything above
        # it gets swallowed into the header band.
        if len(spans) >= MIN_CELLS_PER_ROW and _numfrac(spans) > 0.5
    ]
    return min(ys) if ys else None


# --------------------------------------------------------------------------- #
# grid candidates
# --------------------------------------------------------------------------- #
def _grid_span(data_rows, bucket: str):
    """Span-structure grid: K = modal span count; cuts = midpoints of the
    per-column median x0 over exact-K anchor rows."""
    if len(data_rows) < MIN_TABLE_ROWS:
        return None
    k = Counter(len(r) for r in data_rows).most_common(1)[0][0]
    if k < 2:
        return None
    anchors = [r for r in data_rows if len(r) == k]
    if len(anchors) < 3:
        return None
    col_x = [median(a[j][1] for a in anchors) for j in range(k)]
    cut_xs = [(col_x[j] + col_x[j + 1]) / 2 for j in range(k - 1)]

    grid, counts = [], []
    for row in data_rows:
        cells, cnts = [""] * k, [0] * k
        for t, x0, x1 in row:
            pos = x0 if bucket == "x0" else (x0 + x1) / 2
            c = min(sum(1 for cx in cut_xs if pos >= cx), k - 1)
            cells[c] = f"{cells[c]} {t}".strip() if cells[c] else t
            cnts[c] += 1
        grid.append(cells)
        counts.append(cnts)
    return k, cut_xs, grid, counts


def _grid_proj(data_rows, frac: float):
    """Whitespace-gap projection grid: an x position belongs to a column iff
    covered by a span in > frac of rows."""
    if len(data_rows) < MIN_TABLE_ROWS:
        return None
    lo = min(x0 for r in data_rows for _, x0, _ in r)
    hi = max(x1 for r in data_rows for _, _, x1 in r)
    n = int(hi - lo) + 2
    cov = [0] * n
    for r in data_rows:
        for _, x0, x1 in r:
            for i in range(int(x0 - lo), min(int(x1 - lo) + 1, n)):
                cov[i] += 1
    thr = frac * len(data_rows)
    runs, i = [], 0
    while i < n:
        if cov[i] > thr:
            j = i
            while j < n and cov[j] > thr:
                j += 1
            runs.append((lo + i, lo + j))
            i = j
        else:
            i += 1
    if not runs:
        return None
    cols = [runs[0]]
    for c in runs[1:]:
        if c[0] - cols[-1][1] < GAP_MIN_PT:
            cols[-1] = (cols[-1][0], c[1])
        else:
            cols.append(c)
    k = len(cols)
    if k < 2:
        return None
    cut_xs = [(cols[j][1] + cols[j + 1][0]) / 2 for j in range(k - 1)]

    grid, counts = [], []
    for row in data_rows:
        cells, cnts = [""] * k, [0] * k
        for t, x0, x1 in row:
            best, bo = None, 0.0
            for j, (c0, c1) in enumerate(cols):
                o = max(0.0, min(x1, c1) - max(x0, c0))
                if o > bo:
                    best, bo = j, o
            if best is None:
                cx = (x0 + x1) / 2
                best = min(
                    range(k), key=lambda j: abs((cols[j][0] + cols[j][1]) / 2 - cx)
                )
            cells[best] = f"{cells[best]} {t}".strip() if cells[best] else t
            cnts[best] += 1
        grid.append(cells)
        counts.append(cnts)
    return k, cut_xs, grid, counts


def _candidates(data_rows) -> dict:
    out = {}
    for bucket in ("x0", "center"):
        g = _grid_span(data_rows, bucket)
        if g:
            out[f"span-{bucket}"] = g
    for frac in PROJ_FRACS:
        g = _grid_proj(data_rows, frac)
        if g:
            out[f"proj-{int(frac * 100)}"] = g
    return out


# --------------------------------------------------------------------------- #
# deterministic judge
# --------------------------------------------------------------------------- #
def _cell_class(c: str) -> str:
    c = c.strip()
    if not c:
        return "empty"
    if _YEAR.match(c):
        return "year"
    if _NUMISH.match(c):
        return "num"
    if _PCTISH.search(c):
        return "range"
    return "text"


def _score_grid(k, grid, counts, n_header_cells) -> float:
    """Composite in [0,1]: type purity, compound-cell penalty, fill,
    header-count agreement, spans-per-cell (double weight)."""
    nonempty = [c for r in grid for c in r if c.strip()]
    if not nonempty:
        return 0.0
    fill = len(nonempty) / max(k * len(grid), 1)
    compound = sum(
        1
        for c in nonempty
        if len(_NUM_TOK.findall(c)) >= 2
        and _cell_class(c) == "text"
        and sum(ch.isdigit() for ch in c) > len(c) * 0.4
    )
    no_compound = 1 - compound / len(nonempty)
    purs = []
    for j in range(k):
        cc = Counter(_cell_class(r[j]) for r in grid)
        cc.pop("empty", None)
        if cc:
            purs.append(cc.most_common(1)[0][1] / sum(cc.values()))
    purity = sum(purs) / len(purs) if purs else 0.0
    hdr = 0.5 if n_header_cells is None else max(0.0, 1 - abs(n_header_cells - k) / k)
    occ = [c for row in counts for c in row if c > 0]
    one_span = sum(1 for c in occ if c == 1) / max(len(occ), 1)
    return (purity + no_compound + fill + hdr + 2 * one_span) / 6


def _pick_winner(cands: dict, n_header_cells):
    best_name, best, best_score = None, None, -1.0
    for name, (k, cut_xs, grid, counts) in cands.items():
        s = _score_grid(k, grid, counts, n_header_cells)
        if s > best_score:
            best_name, best, best_score = name, (k, cut_xs, grid, counts), s
    return best_name, best, best_score


# --------------------------------------------------------------------------- #
# headers + wrapped lines
# --------------------------------------------------------------------------- #
def _full_intervals(cut_xs: list) -> list:
    edges = [-1e9, *cut_xs, 1e9]
    return list(itertools.pairwise(edges))


def _avg_col_width(cut_xs: list) -> float:
    widths = [b - a for a, b in itertools.pairwise(cut_xs)]
    return (sum(widths) / len(widths)) if widths else 100.0


def _stitch_band_headers(band_lines, cut_xs, k):
    """Assign header-band spans to columns by center within full-width column
    intervals; stitch top-to-bottom. Returns (names, n_named)."""
    full = _full_intervals(cut_xs)
    avg_w = _avg_col_width(cut_xs)
    parts = defaultdict(list)
    for spans, _y0 in sorted(band_lines, key=lambda t: t[1]):
        for t, x0, x1 in spans:
            if (x1 - x0) > 2.5 * avg_w:
                continue
            cx = (x0 + x1) / 2
            j = next((j for j, (f0, f1) in enumerate(full) if f0 <= cx < f1), None)
            if j is not None:
                parts[j].append(t)
    # Unfilled header columns are left blank rather than "column_N" - a blank
    # <th> reads cleaner than a placeholder label in the output.
    names = [" ".join(parts[j]) if parts.get(j) else "" for j in range(k)]
    return names, sum(1 for j in range(k) if parts.get(j))


def _attach_wrapped_lines(lines, first_data_y, grid_y, cut_xs):
    """Merge short mid-table text lines (wrapped cell continuations) into the
    row above, in the aligned text column. Deterministic merge-up (the pre-LLM
    fallback): numeric spans and non-text columns are never attached."""
    full = _full_intervals(cut_xs)
    avg_w = _avg_col_width(cut_xs)
    k = len(cut_xs) + 1
    col_texty = []
    for j in range(k):
        vals = [cells[j] for cells, _ in grid_y if cells[j].strip()]
        texty = sum(1 for v in vals if _cell_class(v) == "text")
        col_texty.append(bool(vals) and texty >= 0.5 * len(vals))

    for spans, y0, _y1 in sorted(lines, key=lambda t: t[1]):
        if y0 < first_data_y or len(spans) >= MIN_CELLS_PER_ROW:
            continue
        above = [i for i, g in enumerate(grid_y) if g[1] <= y0]
        if not above:
            continue
        ti = above[-1]
        for t, x0, x1 in spans:
            if (x1 - x0) > 2.5 * avg_w:
                continue
            cx = (x0 + x1) / 2
            j = next((j for j, (f0, f1) in enumerate(full) if f0 <= cx < f1), None)
            if j is None or not col_texty[j] or _cell_class(t) != "text":
                continue
            target = grid_y[ti][0]
            target[j] = f"{target[j]} {t}".strip() if target[j] else t


_SYMBOL_ONLY = re.compile(r"^[^\w]{1,2}$")


def _merge_marker_columns(names, grid):
    """Merge symbol-marker columns into their right neighbor.

    A column whose every non-empty DATA value is a short non-alphanumeric
    symbol (checkboxes, bullets, tick marks) is a row *marker*, not a data
    column - keeping it separate splits "(checkbox) CODE" into two cells and
    shifts the header row out of alignment with the data columns. Runs after
    header extraction so the header row doesn't mask an all-symbol column."""
    if not grid:
        return names, grid
    k = len(grid[0])
    j = 0
    while k > 1 and j < k - 1:
        vals = [row[j] for row in grid if row[j].strip()]
        if len(vals) >= 2 and all(_SYMBOL_ONLY.match(v.strip()) for v in vals):
            for row in grid:
                row[j : j + 2] = [f"{row[j]} {row[j + 1]}".strip()]
            if names:
                names[j : j + 2] = [f"{names[j]} {names[j + 1]}".strip()]
            k -= 1
        else:
            j += 1
    return names, grid


def _build_html(names, grid, has_header: bool) -> str:
    out = ["<table>"]
    if has_header:
        out.append("<thead><tr>")
        out += [f"<th>{escape(n)}</th>" for n in names]
        out.append("</tr></thead>")
    out.append("<tbody>")
    for cells in grid:
        out.append("<tr>" + "".join(f"<td>{escape(c)}</td>" for c in cells) + "</tr>")
    out.append("</tbody></table>")
    return "".join(out)


# --------------------------------------------------------------------------- #
# sparse datasheet tables
# --------------------------------------------------------------------------- #
def _cluster_overlapping_spans(lines):
    """Group vertically wrapped spans that occupy the same x interval."""
    spans = sorted(
        [
            (text.strip(), x0, x1, y0)
            for row, y0, _ in lines
            for text, x0, x1 in row
            if text.strip()
        ],
        key=lambda span: (span[1], span[2], span[3]),
    )
    clusters = []
    for span in spans:
        matching = [
            cluster
            for cluster in clusters
            if span[1] <= cluster["x1"] + 1.0 and span[2] >= cluster["x0"] - 1.0
        ]
        if not matching:
            clusters.append({"x0": span[1], "x1": span[2], "spans": [span]})
            continue

        cluster = matching[0]
        cluster["x0"] = min(cluster["x0"], span[1])
        cluster["x1"] = max(cluster["x1"], span[2])
        cluster["spans"].append(span)
        for extra in matching[1:]:
            cluster["x0"] = min(cluster["x0"], extra["x0"])
            cluster["x1"] = max(cluster["x1"], extra["x1"])
            cluster["spans"].extend(extra["spans"])
            clusters.remove(extra)
    return sorted(clusters, key=lambda cluster: cluster["x0"])


def _reconstruct_bitfield_html(lines, bbox):
    """Reconstruct a one-row register bitfield, including wrapped labels."""
    bx0, by0, bx1, by1 = bbox
    if bx1 <= bx0 or by1 <= by0 or (bx1 - bx0) / (by1 - by0) < 5:
        return None

    caption_lines = [
        row
        for row, _, _ in lines
        if len(row) == 1 and re.match(r"^Table\s+\d+\.", row[0][0])
    ]
    bitfield_lines = [
        line
        for line in lines
        if not (len(line[0]) == 1 and re.match(r"^Table\s+\d+\.", line[0][0][0]))
    ]
    clusters = _cluster_overlapping_spans(bitfield_lines)
    if not 3 <= len(clusters) <= 32:
        return None

    centers = [(cluster["x0"] + cluster["x1"]) / 2 for cluster in clusters]
    gaps = [right - left for left, right in itertools.pairwise(centers)]
    typical_gap = median(gaps)
    if typical_gap <= 0:
        return None
    if any(abs(gap - typical_gap) > typical_gap * 0.5 for gap in gaps):
        return None
    if centers[-1] - centers[0] < (bx1 - bx0) * 0.55:
        return None

    cells = []
    for cluster in clusters:
        parts = sorted(cluster["spans"], key=lambda span: (span[3], span[1]))
        if any(
            re.search(r"\s", part[0]) and any(char.islower() for char in part[0])
            for part in parts
        ):
            return None
        if len(parts) > 1 and any(
            any(char.islower() for char in part[0]) for part in parts
        ):
            return None
        cell = "".join(re.sub(r"\s+", "", part[0]) for part in parts)
        if not cell or not _CODE_CELL.match(cell):
            return None
        cells.append(cell)

    html = _build_html([], [cells], False)
    if caption_lines:
        caption = " ".join(row[0][0] for row in caption_lines)
        html = html.replace("<table>", f"<table><caption>{escape(caption)}</caption>", 1)
    return html


def _join_inline_spans(spans):
    text = ""
    previous_x1 = None
    for value, x0, x1 in sorted(spans, key=lambda span: span[1]):
        value = " ".join(value.split())
        if not value:
            continue
        separator = " "
        if previous_x1 is None:
            separator = ""
        elif value[0] in ".,;:!?)]}/" or text[-1] in "([{/_":
            separator = ""
        elif value == "g" and text.endswith(("µ", "m")):
            separator = ""
        text += separator + value
        previous_x1 = x1
    return text


def _reconstruct_definition_html(lines, bbox):
    """Reconstruct compact field/description tables by their x/y anchors."""
    spans = [
        (text.strip(), x0, x1, y0, y1)
        for row, y0, y1 in lines
        for text, x0, x1 in row
        if text.strip()
    ]
    if len(spans) < 2:
        return None

    bx0, _, bx1, _ = bbox
    left_x = min(span[1] for span in spans)
    left_tolerance = max(3.0, (bx1 - bx0) * 0.03)
    keys = []
    for span in spans:
        normalized = re.sub(r"\s+", "", span[0])
        spaced_key = bool(re.search(r"\s", span[0]))
        if (
            span[1] <= left_x + left_tolerance
            and _CODE_CELL.match(normalized)
            and (not spaced_key or span[0].upper() == span[0])
        ):
            keys.append((normalized, *span[1:]))
    if not keys:
        return None

    keys = sorted(keys, key=lambda span: (span[3] + span[4]) / 2)
    right_start = max(span[2] for span in keys) + 3.0
    description_lines = []
    for row, y0, y1 in lines:
        right_spans = [span for span in row if span[1] >= right_start]
        if right_spans:
            gaps = [
                right[1] - left[2]
                for left, right in itertools.pairwise(right_spans)
            ]
            if any(gap > max(3.0, (bx1 - bx0) * 0.01) for gap in gaps):
                return None
            description_lines.append(
                (_join_inline_spans(right_spans), y0, y1, right_spans[0][1])
            )
    if not description_lines:
        return None
    if not any(any(char.islower() for char in line[0]) for line in description_lines):
        return None

    starts = [line[3] for line in description_lines]
    if max(starts) - min(starts) > max(8.0, (bx1 - bx0) * 0.03):
        return None

    descriptions = [[] for _ in keys]
    key_centers = [(span[3] + span[4]) / 2 for span in keys]
    for text, y0, y1, _ in description_lines:
        center = (y0 + y1) / 2
        row_idx = min(
            range(len(keys)), key=lambda idx: abs(key_centers[idx] - center)
        )
        descriptions[row_idx].append((y0, text))
    if any(not parts for parts in descriptions):
        return None

    grid = []
    for key, parts in zip(keys, descriptions, strict=True):
        description = " ".join(text for _, text in sorted(parts))
        grid.append([key[0], description])
    return _build_html([], grid, False)


def reconstruct_sparse_table_html(lines, bbox):
    """Return high-confidence HTML for sparse datasheet tables, or None."""
    if not lines or not _garble_ok(lines):
        return None
    return _reconstruct_bitfield_html(lines, bbox) or _reconstruct_definition_html(
        lines, bbox
    )


def _line_tokens(line: dict, bbox):
    """Tokenize a pdftext line into (text, x0, x1) cells inside ``bbox``.

    Prefer WORD-level tokens split on intra-line character gaps: pdftext often
    merges adjacent table cells into one span, which collapses the column
    structure the grid heuristics rely on. Re-splitting the chars at gaps
    wider than ~a quarter of the line height recovers per-cell tokens. Falls
    back to raw spans when char data isn't kept."""
    bx0, by0, bx1, by1 = bbox

    def inside(x0, y0, x1, y1):
        return bx0 <= (x0 + x1) / 2 <= bx1 and by0 <= (y0 + y1) / 2 <= by1

    tokens = []
    have_chars = False
    for s in line.get("spans", []):
        chars = s.get("chars") or []
        if not chars:
            continue
        have_chars = True
        cur = None  # [text, x0, x1]
        for c in chars:
            cx0, cy0, cx1, cy1 = c["bbox"]
            if not inside(cx0, cy0, cx1, cy1):
                continue
            ch = c.get("char", "")
            # Split threshold: half the char height. 0.25x mis-split words in
            # condensed fonts (letter gaps can reach ~0.3x height); real cell
            # and column gaps sit well above 0.5x.
            gap = 0.5 * max(cy1 - cy0, 1.0)
            if cur is None:
                cur = [ch, cx0, cx1]
            elif cx0 - cur[2] > gap:
                tokens.append(cur)
                cur = [ch, cx0, cx1]
            else:
                cur[0] += ch
                cur[2] = cx1
        if cur is not None:
            tokens.append(cur)

    if not have_chars:
        # No char data: fall back to span-level tokens.
        for s in line.get("spans", []):
            t = (s.get("text") or "").strip()
            sx0, sy0, sx1, sy1 = s["bbox"]
            if t and inside(sx0, sy0, sx1, sy1):
                tokens.append([t, sx0, sx1])

    out = []
    for text, x0, x1 in tokens:
        text = text.strip()
        if text and not _LEADER_ONLY.match(text):
            out.append((text, round(x0, 1), round(x1, 1)))
    return out


def table_lines_from_pdftext(pdftext_page: dict, bbox) -> list:
    """Extract ``[(tokens, y0, y1)]`` lines (tokens = ``[(text, x0, x1)]``) from a
    cached pdftext page, restricted to ``bbox`` (x0, y0, x1, y1, in pdftext/PDF
    points). Tokens are word-level (see _line_tokens). Feeds
    reconstruct_table_html."""
    bx0, by0, bx1, by1 = bbox
    lines = []
    for block in pdftext_page.get("blocks", []):
        for line in block.get("lines", []):
            lb = line.get("bbox")
            # Cheap reject: _line_tokens only keeps chars whose center lies in
            # the table bbox, and every char is contained in the line bbox, so a
            # line whose bbox is disjoint from the table region can contribute
            # nothing. Skipping it avoids re-scanning the whole page's char
            # layer once per table (was O(tables x page_chars)). Lines without a
            # bbox fall through to the full per-char check.
            if lb is not None and (
                lb[2] < bx0 or lb[0] > bx1 or lb[3] < by0 or lb[1] > by1
            ):
                continue
            toks = _line_tokens(line, bbox)
            if toks:
                lb = lb or [0, 0, 0, 0]
                lines.append((toks, round(lb[1], 1), round(lb[3], 1)))
    return lines


def reconstruct_table_html(lines):
    """Reconstruct ``(html, score)`` from a table's pdftext lines, or None.

    ``lines``: [(spans, y0, y1)], spans = [(text, x0, x1)] in PDF points.
    """
    if not lines or not _garble_ok(lines):
        return None

    first_data_y = _find_header_band(lines)
    if first_data_y is None:
        # No numeric transition to mark the header. Fall back to the first
        # span-rich data row: sparse lines above it are a (possibly multi-line)
        # header band, so their text is captured instead of dropped.
        rich_ys = [y0 for spans, y0, _ in lines if len(spans) >= MIN_CELLS_PER_ROW]
        if rich_ys:
            alt_y = min(rich_ys)
            if any(y0 < alt_y for _, y0, _ in lines):
                first_data_y = alt_y
    if first_data_y is not None:
        data = [
            (spans, y0)
            for spans, y0, _ in lines
            if y0 >= first_data_y and len(spans) >= MIN_CELLS_PER_ROW
        ]
        band = [(spans, y0) for spans, y0, _ in lines if y0 < first_data_y]
        n_hdr = (
            len([1 for spans, _ in band for t, x0, x1 in spans if (x1 - x0) < 200])
            or None
        )
    else:
        data = [
            (spans, y0) for spans, y0, _ in lines if len(spans) >= MIN_CELLS_PER_ROW
        ]
        band, n_hdr = [], None

    data_rows = [spans for spans, _ in data]
    cands = _candidates(data_rows)
    if not cands:
        return None
    _name, best, score = _pick_winner(cands, n_hdr)
    if not best:
        return None
    k, cut_xs, grid, counts = best

    if band:
        names, n_named = _stitch_band_headers(band, cut_xs, k)
        has_header = True
    else:
        # No geometric header band: treat the first data row as the header
        # (the common header-in-first-row case); avoids emitting column_N junk.
        names, has_header = [], False

    if len(grid) < MIN_TABLE_ROWS:
        return None

    if first_data_y is not None:
        grid_y = list(zip(grid, [y0 for _, y0 in data[-len(grid) :]]))
        _attach_wrapped_lines(lines, first_data_y, grid_y, cut_xs)
        grid = [g for g, _ in grid_y]

    if not has_header and grid:
        names, grid, has_header = grid[0], grid[1:], True
        if len(grid) < 1:
            return None

    names, grid = _merge_marker_columns(names, grid)

    return _build_html(names, grid, has_header), score
