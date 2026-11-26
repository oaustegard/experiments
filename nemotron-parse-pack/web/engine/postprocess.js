// Tokenizer decode and output postprocessing for Nemotron Parse 2.0: JS ports of the model repo's
// postprocessing.py (extract_classes_bboxes, transform_bbox_to_original, postprocess_text) and the
// parts of latex2html.py they use (latex_table_to_html, the HTML table -> markdown conversion).

// ---- tokenizer (the pack's slim tokenizer.json: {tokens: [...], special: [ids]}) ----

// GPT-2 byte-level alphabet: printable bytes map to themselves, the rest to U+0100 upward.
const CHAR_BYTE = (() => {
  const keep = [];
  for (let b = 33; b <= 126; b++) keep.push(b);
  for (let b = 161; b <= 172; b++) keep.push(b);
  for (let b = 174; b <= 255; b++) keep.push(b);
  const map = new Map(keep.map((b) => [String.fromCharCode(b), b]));
  let n = 0;
  for (let b = 0; b < 256; b++) if (!map.has(String.fromCharCode(b))) map.set(String.fromCharCode(256 + n++), b);
  return map;
})();
const UTF8 = new TextEncoder();

export class Tokenizer {
  constructor(json) {
    this.tokens = json.tokens;
    this.special = new Set(json.special);
    this.bytes = new Array(this.tokens.length);
    this.control = null;
  }

  // Bytes of one token, as tokenizers' ByteLevel decoder gives them: every char through the byte
  // alphabet, or the token's own UTF-8 when any char is outside it (added tokens such as CJK characters).
  tokenBytes(id) {
    let b = this.bytes[id];
    if (b) return b;
    const t = this.tokens[id] ?? "";
    const out = new Uint8Array(t.length);
    let ok = true, i = 0;
    for (const ch of t) {
      const v = CHAR_BYTE.get(ch);
      if (v === undefined) { ok = false; break; }
      out[i++] = v;
    }
    b = ok ? out.subarray(0, i) : UTF8.encode(t);
    this.bytes[id] = b;
    return b;
  }

  decode(ids, skipSpecial = true) {
    const parts = [];
    let n = 0;
    for (const id of ids) {
      if (skipSpecial && this.special.has(id)) continue;
      const b = this.tokenBytes(id);
      parts.push(b);
      n += b.length;
    }
    const all = new Uint8Array(n);
    let at = 0;
    for (const b of parts) { all.set(b, at); at += b.length; }
    return new TextDecoder().decode(all);
  }

  // Incremental decode: push(id) returns the text completed by that token.
  stream(skipSpecial = true) {
    const dec = new TextDecoder();
    return { push: (id) => (skipSpecial && this.special.has(id) ? "" : dec.decode(this.tokenBytes(id), { stream: true })) };
  }

  // Prompts are sequences of control tokens (`</s><s><predict_bbox>...`); free text is not supported.
  encodePrompt(text) {
    if (!this.control) {
      this.control = new Map();
      this.tokens.forEach((t, i) => { if (/^(<[^<>]+>|\[[A-Z_]+\])$/.test(t) && !this.control.has(t)) this.control.set(t, i); });
    }
    const ids = [];
    let at = 0;
    for (const m of text.matchAll(/<[^<>]+>|\[[A-Z_]+\]/g)) {
      if (m.index !== at || !this.control.has(m[0])) throw new Error("prompt must be a sequence of control tokens, got " + JSON.stringify(text.slice(at, m.index + m[0].length)));
      ids.push(this.control.get(m[0]));
      at = m.index + m[0].length;
    }
    if (at !== text.length) throw new Error("prompt must be a sequence of control tokens, got " + JSON.stringify(text.slice(at)));
    return ids;
  }
}

// ---- postprocessing.py ----

export function extractClassesBboxes(text) {
  const re = /<x_(\d+(?:\.\d+)?)><y_(\d+(?:\.\d+)?)>(.*?)<x_(\d+(?:\.\d+)?)><y_(\d+(?:\.\d+)?)><class_([^>]+)>/gs;
  const out = [];
  for (const m of text.matchAll(re)) {
    out.push({ cls: m[6] === "Inline-formula" ? "Formula" : m[6], bbox: [+m[1], +m[2], +m[4], +m[5]], text: m[3] });
  }
  return out;
}

// The processor's resize (shrink only, aspect kept, truncating) for an image of w x h on a targetW x targetH canvas.
export function resizedSize(w, h, targetW = 1664, targetH = 2048) {
  const aspect = w / h;
  let nh = h, nw = w;
  if (h > targetH) { nh = targetH; nw = Math.trunc(nh * aspect); }
  if (nw > targetW) { nw = targetW; nh = Math.trunc(nw / aspect); }
  return [nw, nh];
}

// Normalized canvas box -> pixels of the original image.
export function transformBboxToOriginal(bbox, w, h, targetW = 1664, targetH = 2048) {
  const [rw, rh] = resizedSize(w, h, targetW, targetH);
  const padL = Math.floor(Math.max(0, targetW - rw) / 2), padT = Math.floor(Math.max(0, targetH - rh) / 2);
  const cx = (v) => Math.max(0, Math.min(w, (v * targetW - padL) * w / rw));
  const cy = (v) => Math.max(0, Math.min(h, (v * targetH - padT) * h / rh));
  return [cx(bbox[0]), cy(bbox[1]), cx(bbox[2]), cy(bbox[3])];
}

export function mmdToPlain(t) {
  t = t.replace(/<sup>(.*?)<\/sup>/gs, "^{$1}").replace(/<sub>(.*?)<\/sub>/gs, "_{$1}").replaceAll("<br>", "\n");
  t = t.replace(/#+\s/g, "").replace(/\*\*(.*?)\*\*/g, "$1").replace(/\*(.*?)\*/g, "$1").replace(/(?<!\w)_([^_]+)_/g, "$1");
  return t.trim();
}

// opts: textFormat "markdown" | "plain"; tableFormat "markdown" | "html" | "latex"; blankTextInFigures.
export function postprocessText(text, cls = "Text", opts = {}) {
  const { textFormat = "markdown", tableFormat = "markdown", blankTextInFigures = false } = opts;
  if (cls !== "Table") {
    if (textFormat === "plain") text = mmdToPlain(text);
  } else if (tableFormat !== "latex") {
    try { text = latexTables(text, tableFormat === "markdown" ? "markdown" : "html"); } catch (e) { /* malformed LaTeX: keep it */ }
  }
  if (blankTextInFigures && cls === "Picture") text = "";
  return text;
}

// ---- latex2html.py ----

function parseBraced(s, i) {   // s[i] === "{": [content, index after the matching "}"]
  if (s[i] !== "{") throw new Error("expected { at " + i);
  let depth = 1, j = i + 1;
  for (; j < s.length && depth > 0; j++) {
    if (s[j] === "{") depth++;
    else if (s[j] === "}") depth--;
  }
  if (depth) throw new Error("unbalanced braces at " + i);
  return [s.slice(i + 1, j - 1), j];
}

const skipWs = (s, i) => { while (i < s.length && /\s/.test(s[i])) i++; return i; };

// Escape unescaped & inside the third argument of \multirow / \multicolumn, recursively.
function cleanMultiCells(text) {
  let out = "", i = 0;
  while (i < text.length) {
    const a = text.indexOf("\\multirow", i), b = text.indexOf("\\multicolumn", i);
    if (a === -1 && b === -1) { out += text.slice(i); break; }
    const at = a === -1 ? b : b === -1 ? a : Math.min(a, b);
    out += text.slice(i, at);
    const name = text.startsWith("\\multirow", at) ? "\\multirow" : "\\multicolumn";
    let j = skipWs(text, at + name.length);
    const args = [];
    for (let k = 0; k < 3; k++) {
      let arg;
      [arg, j] = parseBraced(text, j);
      arg = cleanMultiCells(arg);
      if (k === 2) arg = arg.replace(/(?<!\\)&/g, "\\&");
      args.push(arg);
      if (k < 2) j = skipWs(text, j);
    }
    out += `${name}{${args[0]}}{${args[1]}}{${args[2]}}`;
    i = j;
  }
  return out;
}

// A \multirow / \multicolumn command at s[pos], nested commands merged (outer rowspan + inner colspan).
function parseMerge(s, pos) {
  const row = s.startsWith("\\multirow", pos), col = s.startsWith("\\multicolumn", pos);
  if (!row && !col) return [null, pos];
  let j = pos + (row ? 9 : 12), a, b, content;
  [a, j] = parseBraced(s, j);
  [b, j] = parseBraced(s, j);
  [content, j] = parseBraced(s, j);
  const own = row ? { rowspan: a.trim(), width: b.trim() } : { colspan: a.trim(), alignment: b.trim() };
  const idx = [content.indexOf("\\multirow"), content.indexOf("\\multicolumn")].filter((x) => x !== -1);
  if (!idx.length) return [{ ...own, content: content.trim() }, j];
  const [inner] = parseMerge(content, Math.min(...idx));
  return [{ ...own, ...inner }, j];
}

function firstMerged(s) {
  for (let pos = 0; pos < s.length; pos++) {
    if (s[pos] !== "\\") continue;
    const [res] = parseMerge(s, pos);
    if (res) return res;
  }
  return null;
}

function processCell(cell) {
  cell = cell.trim();
  const m = firstMerged(cell);
  if (m) return { content: processCell(m.content).content, colspan: parseInt(m.colspan ?? "1", 10) || 1, rowspan: parseInt(m.rowspan ?? "1", 10) || 1 };
  cell = cell.replace(/\$([^$]*)\$/g, "$1").replace(/\\textbf{([^}]*)}/g, "<b>$1</b>").replace(/\\textit{([^}]*)}/g, "<i>$1</i>");
  cell = cell.replace(/(?<!\\)_(.*?)(?<!\\)_/g, (_, c) => `<i>${c.replaceAll("\\_", "_")}</i>`);
  cell = cell.replace(/(?<!\\)\*\*(.*?)(?<!\\)\*\*/g, (_, c) => `<b>${c.replaceAll("\\*", "*")}</b>`);
  for (const [a, b] of [["\\$", "$"], ["\\%", "%"], ["\\#", "#"], ["\\_", "_"], ["\\textasciitilde{}", "~"], ["\\textasciicircum{}", "^"],
    ["\\newline", "\n"], ["\\textless", "<"], ["\\textgreater", ">"], ["\\*", "*"], ["\\backslash", "\\"], ["\\&", "&"], ["<tbc>", ""],
    ["\\unknown", ""], ["\\<|unk|\\>", ""], ["<u>", "<underline>"], ["</u>", "</underline>"]]) cell = cell.replaceAll(a, b);
  return { content: cell, colspan: 1, rowspan: 1 };
}

// One tabular body -> { html, rows: [[{text, colspan, rowspan}]], headerRows }.
function convertTable(content) {
  const all = [];
  for (let row of content.split("\\\\")) {
    if (!row.trim()) continue;
    row = row.trim();
    if (row.includes("\\hline")) {
      row = row.replaceAll("\\hline", "");
      if (!row.trim()) continue;
    }
    let cells = cleanMultiCells(row).split(/(?<!\\)&/).map(processCell);
    let lines = cells.map((c) => c.content.split(/(?:\n|<br\s*\/?>)+/));
    all.push({ cells, lines });
  }
  if (!all.length) return { html: "<table></table>", rows: [], headerRows: 0 };
  const headerRows = Math.max(1, ...all[0].cells.map((c) => c.rowspan));
  for (const r of all) {
    while (r.cells.length) {
      const c = r.cells[r.cells.length - 1];
      if (c.content.trim() !== "" || c.colspan !== 1 || c.rowspan !== 1) break;
      r.cells.pop(); r.lines.pop();
    }
    r.max = r.lines.length ? Math.max(...r.lines.map((l) => l.length)) : 1;
  }
  const html = ["<table>"], rows = [], tracker = new Set();
  let cur = 0;
  for (const { cells, lines, max } of all) {
    for (let li = 0; li < max; li++) {
      const head = cur < headerRows;
      if (cur === 0) html.push(" <thead>");
      if (cur === headerRows) html.push(" <tbody>");
      html.push("  <tr>");
      const out = [];
      let col = 0;
      cells.forEach((cell, ci) => {
        const seg = li < lines[ci].length ? lines[ci][li] : "";
        const attrs = [];
        if (cell.colspan > 1) attrs.push(`colspan="${cell.colspan}"`);
        if (cell.rowspan > 1 && li === 0) {
          attrs.push(`rowspan="${cell.rowspan}"`);
          for (let r = cur + 1; r < cur + cell.rowspan; r++) for (let c = col; c < col + cell.colspan; c++) tracker.add(r + "," + c);
        }
        if ((cell.rowspan > 1 && li > 0) || (tracker.has(cur + "," + col) && seg === "" && cell.colspan === 1 && cell.rowspan === 1)) {
          col += cell.colspan;
          return;
        }
        const tag = head ? "th" : "td";
        html.push(`    <${tag}${attrs.length ? " " + attrs.join(" ") : ""}>${seg}</${tag}>`);
        out.push({ text: seg, colspan: cell.colspan, rowspan: li === 0 ? cell.rowspan : 1 });
        col += cell.colspan;
      });
      html.push("  </tr>");
      rows.push(out);
      if (cur === headerRows - 1) html.push(" </thead>");
      cur++;
    }
  }
  if (cur > headerRows) html.push(" </tbody>");
  html.push("</table>");
  return { html: html.join("\n"), rows, headerRows: Math.min(headerRows, rows.length) };
}

const cellText = (s) => s.replace(/<[^>]+>/g, " ").replace(/&amp;/g, "&").replace(/&lt;/g, "<").replace(/&gt;/g, ">").replace(/[ \t]+/g, " ").trim();

// Spans expanded to a grid; multi-row headers flattened to dot paths; one markdown table.
function tableMarkdown({ rows, headerRows }) {
  if (!rows.length) return "";
  const ncol = Math.max(...rows.map((r) => r.reduce((n, c) => n + c.colspan, 0)));
  const grid = rows.map(() => new Array(ncol).fill(null));
  rows.forEach((row, r) => {
    let c = 0;
    for (const cell of row) {
      while (c < ncol && grid[r][c] !== null) c++;
      if (c >= ncol) break;
      const t = cellText(cell.text);
      for (let dr = 0; dr < cell.rowspan; dr++) for (let dc = 0; dc < cell.colspan; dc++) {
        if (r + dr < grid.length && c + dc < ncol) grid[r + dr][c + dc] = t;
      }
      c += cell.colspan;
    }
  });
  for (const row of grid) for (let i = 0; i < ncol; i++) if (row[i] === null) row[i] = "";
  const paths = [];
  for (let c = 0; c < ncol; c++) {
    const parts = [];
    let prev = null;
    for (let r = 0; r < headerRows; r++) {
      const v = grid[r][c].trim();
      if (v && v !== prev) { parts.push(v); prev = v; }
    }
    paths.push(parts.length ? parts.join(".") : `column_${c}`);
  }
  const esc = (s) => s.replaceAll("|", "\\|");
  const line = (cells) => "| " + cells.map(esc).join(" | ") + " |";
  return [line(paths), line(paths.map(() => "---")), ...grid.slice(headerRows).map(line)].join("\n");
}

// Every \begin{tabular}...\end{tabular} in `text` replaced by an HTML table or a markdown table.
export function latexTables(text, format = "html") {
  return text.replace(/\\begin{tabular}{([^}]*)}\s*(.*?)\\end{tabular}/gs, (_, spec, content) => {
    const t = convertTable(content);
    return format === "markdown" ? "\n" + tableMarkdown(t) + "\n" : t.html;
  });
}
export const latexTableToHtml = (text) => latexTables(text, "html");
export const latexTableToMarkdown = (text) => latexTables(text, "markdown");

// raw model output -> [{cls, bbox (original image pixels), text}]
export function toBlocks(raw, width, height, targetW = 1664, targetH = 2048, opts = {}) {
  return extractClassesBboxes(raw).map((b) => ({
    cls: b.cls,
    bbox: transformBboxToOriginal(b.bbox, width, height, targetW, targetH),
    text: postprocessText(b.text, b.cls, opts),
  }));
}
