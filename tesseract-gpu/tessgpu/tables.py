"""Tables from word boxes: find table regions in a page, assign words to cells, emit HTML.

    from tessgpu import tables
    blocks = tables.build(lines)              # lines: [{"box": [...], "words": [{"text","box"}]}]
    html = tables.to_html(blocks)

Input lines are in reading order (tessgpu.layout gives that order). A line is a row candidate
when its words split into at least two segments: a word gap at least GAP_FRAC line heights
wide separates segments, so a single space inside a cell never does.

A table is a run of consecutive row candidates, each within ROW_GAP_FRAC line heights of the
one before it, at least MIN_ROWS long. Its columns are the groups of word extents across all
the run's rows: two words belong to one column when the whitespace between them is narrower
than GAP_FRAC line heights, so a gutter is vertical whitespace that every row of the run
shares. The run is a table only when it has at least MIN_COLS columns. Each word goes to the
column containing its centre; the words of one cell on one line are joined with spaces.

Lines outside tables stay as text blocks. Two text columns read line by line never form a
run, since each line is one column and has no internal gap.

Limits: one line is one row (a cell that wraps onto a second line becomes a second row), and
a sparse row with only one segment ends the run.
"""

import html as _html

GAP_FRAC = 0.9       # word gap, in line heights, that separates cells (or column groups)
ROW_GAP_FRAC = 1.5   # largest vertical gap between rows of one table, in line heights
MIN_ROWS = 3
MIN_COLS = 2
MAX_CELL_WORDS = 5   # median words per non-empty cell; prose lines split into columns run longer


def _h(line):
    return max(1.0, line["box"][3] - line["box"][1])


def _segments(line):
    """Split a line's words into cell segments at gaps of GAP_FRAC line heights."""
    ws = sorted(line["words"], key=lambda w: w["box"][0])
    if not ws:
        return []
    cut = GAP_FRAC * _h(line)
    segs = [[ws[0]]]
    for prev, w in zip(ws, ws[1:]):
        if w["box"][0] - prev["box"][2] >= cut:
            segs.append([w])
        else:
            segs[-1].append(w)
    return segs


def _vgap(a, b):
    """Vertical distance between two lines' boxes, 0 when they overlap."""
    return max(0.0, b["box"][1] - a["box"][3], a["box"][1] - b["box"][3])


def _columns(rows):
    """Column groups over all words of the rows: [(x0, x1), ...] sorted left to right."""
    h_med = _median([_h(r) for r in rows])
    cut = GAP_FRAC * h_med
    spans = sorted((w["box"][0], w["box"][2]) for r in rows for w in r["words"])
    groups = []
    for x0, x1 in spans:
        if groups and x0 - groups[-1][1] < cut:
            groups[-1][1] = max(groups[-1][1], x1)
        else:
            groups.append([x0, x1])
    return [(g[0], g[1]) for g in groups]


def _median(vals):
    s = sorted(vals)
    n = len(s)
    if n == 0:
        return 1.0
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2.0


def _union(boxes):
    return [min(b[0] for b in boxes), min(b[1] for b in boxes),
            max(b[2] for b in boxes), max(b[3] for b in boxes)]


def _table(rows):
    """Build a table block from consecutive row lines, or None when they are not a table."""
    cols = _columns(rows)
    if len(cols) < MIN_COLS:
        return None
    grid = []
    counts = []
    for r in rows:
        cells = [[] for _ in cols]
        for w in r["words"]:
            cx = (w["box"][0] + w["box"][2]) / 2.0
            k = _nearest_column(cols, cx)
            cells[k].append(w)
        counts += [len(c) for c in cells if c]
        grid.append([" ".join(w["text"] for w in sorted(c, key=lambda w: w["box"][0]))
                     for c in cells])
    if _median(counts) > MAX_CELL_WORDS:
        return None
    return {
        "kind": "table",
        "box": _union([r["box"] for r in rows]),
        "cols": [[round(a, 2), round(b, 2)] for a, b in cols],
        "rows": grid,
        "line_ids": [r["id"] for r in rows],
    }


def _nearest_column(cols, cx):
    for k, (x0, x1) in enumerate(cols):
        if cx <= x1:
            return k
    return len(cols) - 1


def _is_row(line):
    return len(_segments(line)) >= 2


def _overlap_frac(a, b):
    """Shared vertical extent of two boxes as a fraction of the shorter one's height."""
    overlap = min(a["box"][3], b["box"][3]) - max(a["box"][1], b["box"][1])
    return overlap / min(_h(a), _h(b))


def rows_of(lines):
    """Group line fragments into rows: fragments whose vertical bands overlap by half their height.

    The line finder emits one box per cell, and its order is column by column, so the cells of
    one row are scattered. Fragments are therefore grouped by band, top to bottom, and each row
    keeps the smallest reading-order index of its fragments (idx) for the output order.
    """
    order = sorted(range(len(lines)), key=lambda i: (lines[i]["box"][1], lines[i]["box"][0]))
    rows = []
    for i in order:
        ln = lines[i]
        hit = None
        for r in rows[-3:]:
            if _overlap_frac(r, ln) >= 0.5:
                hit = r
                break
        if hit is None:
            rows.append({"box": list(ln["box"]), "words": list(ln["words"]),
                         "idx": ln["idx"], "id": ln["id"]})
        else:
            hit["box"] = _union([hit["box"], ln["box"]])
            hit["words"] = hit["words"] + list(ln["words"])
            hit["idx"] = min(hit["idx"], ln["idx"])
    return rows


def _nearest_column(cols, cx):
    for k, (x0, x1) in enumerate(cols):
        if cx <= x1:
            return k
    return len(cols) - 1


def _is_row(line):
    return len(_segments(line)) >= 2


def _same_row(a, b):
    """Two consecutive line boxes are one row when they share a vertical band and one layout block.

    The line finder emits one box per cell, so the cells of a table row arrive as consecutive
    fragments that overlap vertically. Lines of a paragraph are stacked and do not overlap.
    """
    if "block" in a and "block" in b and a["block"] != b["block"]:
        return False
    overlap = min(a["box"][3], b["box"][3]) - max(a["box"][1], b["box"][1])
    return overlap >= 0.5 * min(_h(a), _h(b))


def merge_rows(lines):
    """Join consecutive fragments that share a vertical band into one row line."""
    out = []
    for ln in lines:
        if out and _same_row(out[-1], ln):
            prev = out[-1]
            prev["box"] = _union([prev["box"], ln["box"]])
            prev["words"] = prev["words"] + list(ln["words"])
        else:
            out.append({"id": ln.get("id"), "box": list(ln["box"]),
                        "words": list(ln["words"]), **({"block": ln["block"]}
                                                       if "block" in ln else {})})
    return out


def build(lines):
    """Blocks in reading order: {"kind": "table", ...} and {"kind": "text", ...}.

    Input lines are in layout reading order, each with "box" and "words" and optionally "id".
    Fragments are grouped into rows (rows_of); consecutive rows that are cell rows form tables;
    every block is then emitted in the reading order of its first fragment.
    """
    frags = []
    for i, ln in enumerate(lines):
        frags.append({"box": ln["box"], "words": ln["words"], "idx": i,
                      "id": ln.get("id", i)})
    rows = rows_of(frags)
    rows.sort(key=lambda r: (r["box"][1], r["box"][0]))
    items = []                           # (reading-order key, block)
    i = 0
    n = len(rows)
    while i < n:
        if not _is_row(rows[i]):
            items.append((rows[i]["idx"], _text_block(rows[i])))
            i += 1
            continue
        j = i + 1
        while j < n and _is_row(rows[j]) and _vgap(rows[j - 1], rows[j]) <= \
                ROW_GAP_FRAC * _median([_h(rows[j - 1]), _h(rows[j])]):
            j += 1
        run = rows[i:j]
        table = _table(run) if len(run) >= MIN_ROWS else None
        if table is not None:
            items.append((min(r["idx"] for r in run), table))
        else:
            for r in run:
                items.append((r["idx"], _text_block(r)))
        i = j
    items.sort(key=lambda t: t[0])
    return [b for _, b in items]


def _text_block(row):
    return {"kind": "text", "box": row["box"], "line_ids": [row["id"]],
            "text": " ".join(w["text"] for w in sorted(row["words"], key=lambda w: w["box"][0]))}


def to_html(blocks):
    """HTML for the blocks: a <table> per table block, one <p> per text line."""
    out = []
    for b in blocks:
        if b["kind"] == "table":
            out.append("<table>")
            for row in b["rows"]:
                out.append("<tr>" + "".join("<td>%s</td>" % _html.escape(c) for c in row) + "</tr>")
            out.append("</table>")
        else:
            out.append("<p>%s</p>" % _html.escape(b["text"]))
    return "\n".join(out)
