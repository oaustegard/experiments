// dizione.js — English text to tiny-voca phonemes in the browser: a JavaScript port of dizione 0.1.0.
//
//   <script src="dizione.js"></script>
//   const dizione = await Dizione.load();                    // or Dizione.load(baseUrl), Dizione.load({ bundle })
//   dizione.phonemize("Hello, it costs $3.50.")              // {ids, stress, text}, null, or throws
//
// dizione is by Muhammed Nazeem (github.com/n4ze3m/dizione), Apache License 2.0; see LICENSE-dizione.txt. This
// file translates its Python source (engine, normalize, convert, morphology, homographs, align, g2p, sentence,
// symbols, tables) line for line, so that every sentence gives the ids and stress Python dizione gives. The data
// bundle (DATA_FILE below) carries dizione's tables and the built data of tiny-voca's phonemizer (dictionary from
// CMUdict, homograph weights from the Wikipedia Homograph Data, the unknown-word model), re-encoded by
// run/dizione_bundle.py. `phonemize` follows the contract tinyvoca.js had with Pyodide: a result, `null` when the
// text has nothing to pronounce, or an exception for any other failure.
//
// Works as a classic script in a page or a Web Worker (defines self.Dizione) and in Node (globalThis.Dizione).
(function (root) {
  "use strict";

  const DATA_FILE = "dizione-da3503a79b1c.bin";            // replaced by run/dizione_bundle.py with the content-named file

  // ---- Python compatibility ------------------------------------------------------------------------------
  // Python's str methods and `re` work on code points with Unicode classes; these helpers give JS the same.

  class ValueError extends Error {}
  class KeyError extends Error {}

  const W = "\\p{L}\\p{N}_";                                   // Python's \w
  const S = "\\t-\\r\\x1c-\\x20\\x85\\xa0\\u{1680}\\u{2000}-\\u{200a}\\u{2028}\\u{2029}\\u{202f}\\u{205f}\\u{3000}";   // \s
  const BOUNDARY = `(?:(?<=[${W}])(?![${W}])|(?<![${W}])(?=[${W}]))`;
  const SYNTAX = "^$\\.*+?()[]{}|/";
  // Python's IGNORECASE: a letter also matches the characters whose lower case is it or its twin (i ı, s ſ, µ μ).
  const CASE_EXTRA = { i: "ıİ", s: "ſ", k: "\u{212a}", "μ": "µΜ", "µ": "μΜ" };

  function caseVariants(ch) {
    const out = new Set([ch]);
    for (const v of [ch.toLowerCase(), ch.toUpperCase()]) if ([...v].length === 1) out.add(v);
    const extra = CASE_EXTRA[ch.toLowerCase()];
    if (extra) for (const v of extra) out.add(v);
    return [...out];
  }
  const isCased = (ch) => ch.toLowerCase() !== ch.toUpperCase();
  const classChar = (ch) => ("\\]^-[".includes(ch) ? "\\" + ch : ch);

  // Python regular expression source -> RegExp. Translates \w \d \s \b, ".", "$", (?P<name>), (?P=name) and
  // (with `ci`) IGNORECASE, which JS would apply with different case rules.
  function rx(src, flags = "", ci = false) {
    let out = "", i = 0, on = ci;
    const stack = [];
    const escape = (inClass) => {
      const c = src[i + 1];
      if (c === "u" && src[i + 2] === "{") { const j = src.indexOf("}", i); const t = src.slice(i, j + 1); i = j + 1; return t; }
      i += 2;
      switch (c) {
        case "w": return inClass ? W : `[${W}]`;
        case "W": if (inClass) throw new Error("\\W in a class"); return `[^${W}]`;
        case "d": return "\\p{Nd}";
        case "D": return "\\P{Nd}";
        case "s": return inClass ? S : `[${S}]`;
        case "S": if (inClass) throw new Error("\\S in a class"); return `[^${S}]`;
        case "b": if (inClass) throw new Error("\\b in a class"); return BOUNDARY;
        default:
          if (/[0-9nrtfvx]/.test(c)) return "\\" + c;
          if (SYNTAX.includes(c) || (inClass && c === "-")) return "\\" + c;
          return inClass ? classChar(c) : c;
      }
    };
    while (i < src.length) {
      const c = src[i];
      if (c === "\\") { out += escape(false); continue; }
      if (c === "[") {
        let body = "", extra = "";
        i++;
        if (src[i] === "^") { body = "^"; i++; }
        // one member of a class: [JS text, the character it stands for or null for \w \d \s]
        const atom = () => {
          if (src[i] === "\\") {
            if (src[i + 1] === "u" && src[i + 2] === "{") {
              const j = src.indexOf("}", i);
              const ch = String.fromCodePoint(parseInt(src.slice(i + 3, j), 16));
              i = j + 1;
              return [classChar(ch), ch];
            }
            const literal = /[wdsWDS0-9nrtfvx]/.test(src[i + 1]) ? null : src[i + 1];
            return [escape(true), literal];
          }
          const ch = String.fromCodePoint(src.codePointAt(i));
          i += ch.length;
          return [classChar(ch), ch];
        };
        while (src[i] !== "]") {
          const [text, ch] = atom();
          if (src[i] === "-" && src[i + 1] !== "]") {
            i++;
            const [hiText, hi] = atom();
            if (ch === null || hi === null) throw new Error("bad range in " + src);
            body += text + "-" + hiText;
            if (on && /[a-z]/.test(ch) && /[a-z]/.test(hi)) extra += ch.toUpperCase() + "-" + hi.toUpperCase() + "ıİſ\u{212a}";
            else if (on && /[A-Z]/.test(ch) && /[A-Z]/.test(hi)) extra += ch.toLowerCase() + "-" + hi.toLowerCase() + "ıİſ\u{212a}";
            continue;
          }
          body += text;
          if (on && ch !== null && isCased(ch)) extra += caseVariants(ch).map(classChar).join("");
        }
        i++;
        out += "[" + body + extra + "]";
        continue;
      }
      if (c === "(") {
        stack.push(on);
        if (src.startsWith("(?P<", i)) { out += "(?<"; i += 4; continue; }
        if (src.startsWith("(?P=", i)) { stack.pop(); const j = src.indexOf(")", i); out += "\\k<" + src.slice(i + 4, j) + ">"; i = j + 1; continue; }
        if (src.startsWith("(?-i:", i)) { out += "(?:"; i += 5; on = false; continue; }
        if (src.startsWith("(?(", i)) throw new Error("conditional groups are not supported");
        out += "("; i++; continue;
      }
      if (c === ")") { on = stack.pop(); out += ")"; i++; continue; }
      if (c === ".") { out += "[^\\n]"; i++; continue; }
      if (c === "$") { out += "(?=\\n?$)"; i++; continue; }
      if (c === "{") {
        const m = /^\{(\d*)(?:(,)(\d*))?\}/.exec(src.slice(i));
        if (m && (m[1] || m[3])) { out += "{" + (m[1] || "0") + (m[2] || "") + (m[3] ?? "") + "}"; i += m[0].length; continue; }
        out += "\\{"; i++; continue;
      }
      if (c === "}") { out += "\\}"; i++; continue; }
      const ch = String.fromCodePoint(src.codePointAt(i));
      i += ch.length;
      if (on && isCased(ch)) { out += "[" + caseVariants(ch).map(classChar).join("") + "]"; continue; }
      out += ch;
    }
    return new RegExp(out, "u" + flags);
  }

  class Match {
    constructor(m, string) { this.m = m; this.string = string; }
    group(g = 0) { const v = typeof g === "number" ? this.m[g] : this.m.groups[g]; return v === undefined ? null : v; }
    start(g = 0) {
      if (g === 0) return this.m.index;
      const at = typeof g === "number" ? this.m.indices[g] : this.m.indices.groups[g];
      return at ? at[0] : -1;
    }
    end(g = 0) {
      if (g === 0) return this.m.index + this.m[0].length;
      const at = typeof g === "number" ? this.m.indices[g] : this.m.indices.groups[g];
      return at ? at[1] : -1;
    }
    groups() { return this.m.slice(1).map((v) => (v === undefined ? null : v)); }
  }

  // The regexes below are compiled with "g" for sub/finditer/findall and "y" for match.
  function sub(re, repl, text) {
    re.lastIndex = 0;
    let out = "", last = 0, m;
    while ((m = re.exec(text)) !== null) {
      out += text.slice(last, m.index) + (typeof repl === "string" ? repl : repl(new Match(m, text)));
      last = m.index + m[0].length;
      if (m[0].length === 0) re.lastIndex += text.codePointAt(re.lastIndex) > 0xffff ? 2 : 1;
    }
    return out + text.slice(last);
  }
  function* finditer(re, text) {
    re.lastIndex = 0;
    let m;
    while ((m = re.exec(text)) !== null) {
      yield new Match(m, text);
      if (m[0].length === 0) re.lastIndex += text.codePointAt(re.lastIndex) > 0xffff ? 2 : 1;
    }
  }
  function findall(re, text) {
    const out = [];
    for (const m of finditer(re, text)) {
      const g = m.m.length - 1;
      out.push(g === 0 ? m.m[0] : g === 1 ? (m.m[1] ?? "") : m.m.slice(1).map((v) => v ?? ""));
    }
    return out;
  }
  function search(re, text) { re.lastIndex = 0; const m = re.exec(text); return m ? new Match(m, text) : null; }
  function match(re, text) { re.lastIndex = 0; const m = re.exec(text); return m ? new Match(m, text) : null; }   // re has "y"
  function split(re, text) {                        // no capture groups in the patterns split here
    const out = [];
    let last = 0;
    for (const m of finditer(re, text)) { out.push(text.slice(last, m.start())); last = m.end(); }
    out.push(text.slice(last));
    return out;
  }
  const R = (src, ci = false) => rx(src, "g", ci);
  const RD = (src, ci = false) => rx(src, "gd", ci);
  const RM = (src, ci = false) => rx(src, "y", ci);
  const RF = (src) => { const re = rx(src, "y"); return new RegExp("(?:" + re.source + ")$", "uy"); };

  const SPACE = new RegExp(`^[${S}]$`, "u");
  const isSpaceCp = (ch) => SPACE.test(ch);
  const ALPHA = /^\p{L}+$/u, ALNUM = /^[\p{L}\p{N}]+$/u, DECIMAL = /^\p{Nd}+$/u, ASCII = /^[\x00-\x7f]*$/;
  // Characters with a digit value that are not decimal digits (superscripts, circled digits, ...): str.isdigit().
  const OTHER_DIGITS = rx("[\\u{b2}\\u{b3}\\u{b9}\\u{1369}-\\u{1371}\\u{19da}\\u{2070}\\u{2074}-\\u{2079}\\u{2080}-\\u{2089}" +
    "\\u{2460}-\\u{2468}\\u{2474}-\\u{247c}\\u{2488}-\\u{2490}\\u{24ea}\\u{24f5}-\\u{24fd}\\u{24ff}\\u{2776}-\\u{277e}" +
    "\\u{2780}-\\u{2788}\\u{278a}-\\u{2792}\\u{10a40}-\\u{10a43}\\u{10e60}-\\u{10e68}\\u{11052}-\\u{1105a}\\u{1f100}-\\u{1f10a}]");
  const DIGIT = new RegExp(`^(?:\\p{Nd}|${OTHER_DIGITS.source})+$`, "u");
  const isalpha = (s) => ALPHA.test(s);
  const isalnum = (s) => ALNUM.test(s);
  const isdigit = (s) => DIGIT.test(s);
  const isascii = (s) => ASCII.test(s);
  const isspace = (s) => s.length > 0 && [...s].every(isSpaceCp);
  const LOWER = /\p{Lowercase}/u, UPPER = /\p{Uppercase}/u, TITLE = /\p{Lt}/u;
  function isupper(s) {
    let cased = false;
    for (const ch of s) {
      if (LOWER.test(ch) || TITLE.test(ch)) return false;
      if (UPPER.test(ch)) cased = true;
    }
    return cased;
  }
  function islower(s) {
    let cased = false;
    for (const ch of s) {
      if (UPPER.test(ch) || TITLE.test(ch)) return false;
      if (LOWER.test(ch)) cased = true;
    }
    return cased;
  }
  const lower = (s) => s.toLowerCase();
  const upper = (s) => s.toUpperCase();
  const plen = (s) => { let n = 0; for (const _ of s) n++; return n; };
  const cpAt = (s, i) => (i < s.length ? String.fromCodePoint(s.codePointAt(i)) : "");
  function cpBefore(s, i) {                        // the character that ends at index i
    if (i <= 0) return "";
    const c = s.charCodeAt(i - 1);
    return c >= 0xdc00 && c <= 0xdfff && i >= 2 ? s.slice(i - 2, i) : s[i - 1];
  }
  function lstrip(s) { let i = 0; while (i < s.length) { const ch = cpAt(s, i); if (!isSpaceCp(ch)) break; i += ch.length; } return s.slice(i); }
  function rstrip(s, chars) {
    let j = s.length;
    while (j > 0) {
      const ch = cpBefore(s, j);
      if (chars === undefined ? !isSpaceCp(ch) : !chars.includes(ch)) break;
      j -= ch.length;
    }
    return s.slice(0, j);
  }
  const strip = (s) => rstrip(lstrip(s));
  function stripChars(s, chars) { let i = 0; while (i < s.length && chars.includes(s[i])) i++; return rstrip(s.slice(i), chars); }
  const WHITESPACE_RUN = new RegExp(`[${S}]+`, "u");
  const words = (s) => strip(s).split(WHITESPACE_RUN).filter((x) => x);           // str.split()
  function cpLast(s, n) { const a = [...s]; return a.slice(Math.max(0, a.length - n)).join(""); }
  function cpBack(s, position, n) {                 // index n characters before position, at least 0
    let i = position;
    while (n > 0 && i > 0) { i -= cpBefore(s, i).length; n--; }
    return i;
  }
  const replaceAll = (s, a, b) => s.split(a).join(b);
  function rpartition(s, sep) { const k = s.lastIndexOf(sep); return k < 0 ? ["", "", s] : [s.slice(0, k), sep, s.slice(k + sep.length)]; }
  function partition(s, sep) { const k = s.indexOf(sep); return k < 0 ? [s, "", ""] : [s.slice(0, k), sep, s.slice(k + sep.length)]; }
  function cmp(a, b) {                              // Python string order (code points)
    const x = [...a], y = [...b];
    for (let k = 0; k < Math.min(x.length, y.length); k++) {
      const d = x[k].codePointAt(0) - y[k].codePointAt(0);
      if (d) return d;
    }
    return x.length - y.length;
  }
  function get(table, key) {                        // dict[key]
    if (!Object.prototype.hasOwnProperty.call(table, key)) throw new KeyError(key);
    return table[key];
  }
  const has = (table, key) => Object.prototype.hasOwnProperty.call(table, key);

  // int() of a string of decimal digits in any script, as a BigInt.
  const ND = /\p{Nd}/u;
  function digitValue(ch) {
    if (ch >= "0" && ch <= "9") return ch.charCodeAt(0) - 48;
    let cp = ch.codePointAt(0);
    if (!ND.test(ch)) throw new ValueError(`invalid literal for int(): ${ch}`);
    let start = cp;
    while (ND.test(String.fromCodePoint(start - 1))) start--;
    return (cp - start) % 10;
  }
  function int(s) {
    s = strip(s);
    if (!s) throw new ValueError("invalid literal for int()");
    let sign = 1n;
    if (s[0] === "+" || s[0] === "-") { if (s[0] === "-") sign = -1n; s = s.slice(1); }
    let n = 0n;
    for (const ch of s) n = n * 10n + BigInt(digitValue(ch));
    if (!s) throw new ValueError("invalid literal for int()");
    return sign * n;
  }

  // Python 3.12+ sum() of floats: Neumaier compensated summation, starting from the integer 0.
  function pysum(values) {
    if (!values.length) return 0;
    let f = 0 + values[0], c = 0;
    for (let k = 1; k < values.length; k++) {
      const x = values[k], t = f + x;
      if (Math.abs(f) >= Math.abs(x)) c += (f - t) + x; else c += (x - t) + f;
      f = t;
    }
    if (c && Number.isFinite(c)) f += c;
    return f;
  }

  // ---- symbols.py --------------------------------------------------------------------------------------
  const PAD = "<pad>", BOS = "<bos>", EOS = "<eos>", SP = "<sp>", UNK = "<unk>";
  const PUNCTUATION = [",", ".", "?", "!", ";", ":", '"', "'", "(", ")", "—", "…"];
  const CONSONANTS = ["p", "b", "t", "d", "k", "ɡ", "f", "v", "θ", "ð", "s", "z", "ʃ", "ʒ", "h",
    "m", "n", "ŋ", "l", "ɹ", "j", "w", "tʃ", "dʒ", "ɾ", "ʔ"];
  const VOWELS = ["ə", "ɐ", "ɚ", "ᵻ", "ɪ", "i", "iː", "ɛ", "æ", "ʌ", "ɑː", "ɔ", "ɔː", "ʊ", "uː", "ɜː",
    "eɪ", "aɪ", "ɔɪ", "oʊ", "aʊ", "iə", "oː",
    "ɑːɹ", "ɔːɹ", "oːɹ", "ɪɹ", "ɛɹ", "ʊɹ", "aɪɚ", "aɪə", "əl", "n̩"];
  const PHONES = CONSONANTS.concat(VOWELS);
  const SYMBOLS = [PAD, BOS, EOS, SP, UNK].concat(PUNCTUATION, PHONES);
  const SYMBOL_TO_ID = new Map(SYMBOLS.map((s, k) => [s, k]));
  const PHONE_SET = new Set(PHONES);
  const VOWEL_SET = new Set(VOWELS);
  const STRESS_MARKS = { 1: "ˈ", 2: "ˌ" };
  const PUNCT_SET = new Set(PUNCTUATION);

  function tokenize(pieces) {
    const symbols = [BOS], stress = [0];
    for (const [punctuation, value] of pieces) {
      if (punctuation) {
        if (symbols[symbols.length - 1] !== value) { symbols.push(value); stress.push(0); }
        continue;
      }
      for (const word of value) {
        const last = symbols[symbols.length - 1];
        if (PHONE_SET.has(last) || last === UNK) { symbols.push(SP); stress.push(0); }
        for (const [phone, level] of word) {
          if (PHONE_SET.has(phone)) { symbols.push(phone); stress.push(level); }
          else { symbols.push(UNK); stress.push(0); }
        }
      }
    }
    symbols.push(EOS);
    stress.push(0);
    if (!symbols.some((s) => PHONE_SET.has(s) || s === UNK)) throw new ValueError("the text has nothing to pronounce");
    return {
      ids: symbols.map((s) => SYMBOL_TO_ID.get(s)),
      stress,
      text: symbols.map((s, k) => (STRESS_MARKS[stress[k]] || "") + s).join(" "),
    };
  }

  // ---- align.py ----------------------------------------------------------------------------------------
  const MAX_RUN = 4, UNSEEN = -14.0;
  const base = (symbol) => symbol.replace(/[012]+$/, "");

  class Aligner {
    constructor(table) {
      this.table = new Map(Object.entries(table).map(([phone, runs]) => [phone, new Map(Object.entries(runs))]));
    }
    score(phone, run) {
      const known = this.table.get(phone);
      const value = known ? known.get(run) : undefined;
      return value !== undefined ? value : UNSEEN - 2.0 * Math.abs(run.length - 1);
    }
    align(word, phones) {
      phones = phones.map(base);
      const letters = word.length, count = phones.length;
      if (!count || letters > MAX_RUN * count) return null;
      const low = -1e18;
      const best = [], step = [];
      for (let j = 0; j <= count; j++) { best.push(new Array(letters + 1).fill(low)); step.push(new Array(letters + 1).fill(0)); }
      best[0][0] = 0.0;
      for (let j = 1; j <= count; j++) {
        const phone = phones[j - 1], previous = best[j - 1], row = best[j], trace = step[j];
        for (let i = 0; i <= letters; i++) {
          for (let k = 0; k <= Math.min(MAX_RUN, i); k++) {
            if (previous[i - k] > low) {
              const value = previous[i - k] + this.score(phone, word.slice(i - k, i));
              if (value > row[i]) { row[i] = value; trace[i] = k; }
            }
          }
        }
      }
      if (best[count][letters] <= low) return null;
      const runs = [];
      let i = letters;
      for (let j = count; j > 0; j--) {
        const k = step[j][i];
        runs.push(word.slice(i - k, i));
        i -= k;
      }
      return runs.reverse();
    }
  }

  // ---- convert.py --------------------------------------------------------------------------------------
  const ARPA_VOWELS = new Set("AA AE AH AO AW AY EH ER EY IH IY OW OY UH UW".split(" "));
  const ARPA_CONSONANTS = { B: "b", CH: "tʃ", D: "d", DH: "ð", F: "f", G: "ɡ", HH: "h", JH: "dʒ", K: "k", L: "l",
    M: "m", N: "n", NG: "ŋ", P: "p", R: "ɹ", S: "s", SH: "ʃ", T: "t", TH: "θ", V: "v", W: "w", Y: "j", Z: "z", ZH: "ʒ" };
  const FULL_VOWELS = { AA: "ɑː", AE: "æ", AH: "ʌ", AO: "ɔː", AW: "aʊ", AY: "aɪ", EH: "ɛ", ER: "ɜː", EY: "eɪ",
    IH: "ɪ", IY: "iː", OW: "oʊ", OY: "ɔɪ", UH: "ʊ", UW: "uː" };
  const ARPABET = new Set(Object.keys(ARPA_CONSONANTS));
  for (const v of ARPA_VOWELS) for (const d of "012") ARPABET.add(v + d);
  const SIBILANTS = new Set(["S", "Z", "SH", "ZH", "CH", "JH"]);
  const VOWEL_LETTERS = new Set("aeiouy");
  const FORCE_SPELLINGS = ["forc", "ford", "sword", "worn", "torn", "sworn", "borne", "shorn", "pork", "divorc"];
  const PREFIXES = ["dis", "mis", "un", "in", "re", "non", "over", "pre"];
  const WEAK_PREFIXES = ["be", "de", "re", "pre"];
  const IRE_ENDINGS = ["ing", "ings", "ingly", "able", "y", "er", "ers", "est"];
  const startsWithAny = (s, list) => list.some((p) => s.startsWith(p));
  const endsWithAny = (s, list) => list.some((p) => s.endsWith(p));

  const isArpabet = (symbols) => symbols.length > 0 && symbols.every((s) => ARPABET.has(s));
  const vowelLetters = (run) => [...run].filter((c) => VOWEL_LETTERS.has(c)).join("");
  const toText = (phones) => phones.map(([phone, stress]) => (STRESS_MARKS[stress] || "") + phone).join(" ");

  function fromText(text) {
    const phones = [];
    for (const unit of words(text)) {
      const stress = unit[0] === "ˈ" ? 1 : unit[0] === "ˌ" ? 2 : 0;
      const phone = stress ? unit.slice(1) : unit;
      if (!PHONE_SET.has(phone)) throw new ValueError(`${phone} is not a phone of the inventory`);
      if (stress && !VOWEL_SET.has(phone)) throw new ValueError(`stress mark on the consonant ${phone}`);
      phones.push([phone, stress]);
    }
    return phones;
  }

  class AlignedWord {
    constructor(word, arpabet, runs) {
      this.word = word;
      this.bases = arpabet.map(base);
      this.stress = arpabet.map((s) => ("012".includes(s[s.length - 1]) ? Number(s[s.length - 1]) : null));
      arpabet.forEach((symbol, k) => {
        const name = this.bases[k];
        if (!has(ARPA_CONSONANTS, name) && !ARPA_VOWELS.has(name)) throw new ValueError(`${symbol} is not an ARPABET symbol`);
      });
      this.aligned = runs !== null && runs !== undefined;
      this.runs = this.aligned ? runs.slice() : arpabet.map(() => "");
      this.starts = [];
      let position = 0;
      for (const run of this.runs) { this.starts.push(position); position += run.length; }
      this.count = this.bases.length;
    }
    base(i) { return i >= 0 && i < this.count ? this.bases[i] : null; }
    vowel(i) { return i >= 0 && i < this.count && ARPA_VOWELS.has(this.bases[i]); }
    unstressed(i) { return this.vowel(i) && !this.stress[i]; }
    letters(i) { return vowelLetters(this.runs.at(i)); }
    before(i) { return this.aligned ? this.word.slice(0, this.starts.at(i)) : ""; }
    after(i) { return this.aligned ? this.word.slice(this.starts.at(i) + this.runs.at(i).length) : ""; }
    endsWith(i, ...symbols) {
      const rest = this.bases.slice(i + 1);
      return rest.length === symbols.length && rest.every((b, k) => b === symbols[k]);
    }
    firstVowel(i) { return !this.bases.slice(0, Math.max(i, 0)).some((b) => ARPA_VOWELS.has(b)); }
    nextVowelStress(i) {
      for (let j = i + 1; j < this.count; j++) if (ARPA_VOWELS.has(this.bases[j])) return this.stress[j];
      return null;
    }
    weakPrefix(i) {
      return this.aligned && this.firstVowel(i) && this.letters(i) === "e" && startsWithAny(this.word, WEAK_PREFIXES)
        && ["b", "d", "r", "pr"].includes(this.before(i));
    }
  }

  function northOrForce(w, i, beforeVowel) {
    if (!w.aligned) return "ɔː";
    const spelled = w.letters(i);
    if (["oa", "oo", "ou"].includes(spelled)) return "oː";
    if (spelled !== "o") return "ɔː";
    const run = w.runs[i];
    const tail = run.slice(run.indexOf("o") + 1) + w.after(i);
    const rest = tail.slice(1);
    if (!beforeVowel) {
      if (rest.startsWith("e")) return "oː";
      if (w.before(i).endsWith("p") && rest.startsWith("t")) return "oː";
      return FORCE_SPELLINGS.some((s) => w.word.includes(s)) ? "oː" : "ɔː";
    }
    if (rest.startsWith("r")) return "ɔː";
    if (["y", "ies", "ied", "a", "as", "ah", "er", "ers"].includes(rest) || startsWithAny(rest, ["ia", "io", "iu", "ing", "age"])) return "oː";
    return "ɔː";
  }

  function weakI(w, i) {
    const nxt = w.base(i + 1), nxt2 = w.base(i + 2);
    if (nxt === "F" && nxt2 === "AH" && w.base(i + 3) === "L") return "i";
    if ((nxt === "T" && nxt2 === "IY") || (nxt === "B" && (nxt2 === "AH" || nxt2 === "L")) || (nxt === "F" && nxt2 === "AY")) return "ᵻ";
    if (nxt === "T" && w.endsWith(i, "T") && ["R", "ER"].includes(w.base(i - 1)) && w.word.endsWith("ite")) return "ᵻ";
    return "ɪ";
  }

  function weakE(w, i) {
    const nxt = w.base(i + 1), prev = w.base(i - 1);
    if (nxt === "T" && (w.endsWith(i, "T") || w.endsWith(i, "T", "S"))) return prev === "L" ? "ə" : prev === "R" ? "ᵻ" : "ɪ";
    if (nxt === "S" && (w.endsWith(i, "S", "T") || w.endsWith(i, "S", "T", "L", "IY"))) return "ɪ";
    if (nxt === "JH") return "ɪ";
    if (nxt === "K" && w.base(i + 2) === "T") return "ɛ";
    if (nxt === "D" && w.endsWith(i, "D")) return "ɪ";
    return "ə";
  }

  function inflection(w, i) {
    if (i + 1 >= w.count || (w.aligned && !(w.runs.at(i) + w.runs.at(i + 1)).includes("e"))) return false;
    const nxt = w.base(i + 1), prev = w.base(i - 1);
    if (nxt === "D" && ["T", "D"].includes(prev) && (w.endsWith(i, "D") || w.endsWith(i, "D", "L", "IY"))) return true;
    return nxt === "Z" && SIBILANTS.has(prev) && w.endsWith(i, "Z");
  }

  function firstSyllable(w, i) {
    if (!w.aligned || !w.firstVowel(i) || !w.nextVowelStress(i)) return null;
    const word = w.word;
    if (i === 0 && word.startsWith("e")) {
      if (word.startsWith("ex")) return "ɛ";
      if (["n", "m", "s"].includes(word.slice(1, 2)) && word.slice(2, 3) && !VOWEL_LETTERS.has(word[2]) && word[2] !== word[1]) return "ɛ";
      return "ɪ";
    }
    if (w.weakPrefix(i)) {
      if (word.startsWith("pre") || (word.startsWith("be") && ["c", "g"].includes(word.slice(2, 3)))) return "ɪ";
      return "ᵻ";
    }
    if (w.letters(i) === "i" && word.startsWith("di") && word.length > 3 && !VOWEL_LETTERS.has(word[2])
        && word[2] !== "s" && VOWEL_LETTERS.has(word[3])) return "ᵻ";
    return null;
  }

  function reduced(w, i) {
    const name = w.bases[i], spelled = w.letters(i);
    const nxt = w.base(i + 1), prev = w.base(i - 1);
    if (name === "IH" && nxt === "NG") return "ɪ";
    if (inflection(w, i)) return "ᵻ";
    if (name === "AH") {
      if (prev === "Y" && (w.runs.at(i - 1) + w.runs.at(i)).includes("u")) return "ʊ";
      if (i === 0 && w.word.startsWith("a")) return "ɐ";
      if (i === 0 && nxt === "N" && w.word.startsWith("un")) return "ʌ";
      if (spelled === "a" && PREFIXES.includes(w.before(i)) && w.runs.at(i).startsWith("a")) return "ɐ";
    }
    const prefix = firstSyllable(w, i);
    if (prefix) return prefix;
    if (spelled === "e" && ["K", "G"].includes(nxt) && w.runs.at(i + 1).startsWith("x") && w.nextVowelStress(i)) return "ɛ";
    if (["i", "ai", "ui", "y"].includes(spelled)) return weakI(w, i);
    if (spelled === "e") return weakE(w, i);
    if (spelled === "a") {
      if (nxt === "JH" || (nxt === "N" && w.base(i + 2) === "JH")) return "ɪ";
      if (nxt === "S" && w.endsWith(i, "S") && w.word.endsWith("ace")) return "ɪ";
      if (name === "IH" && !(nxt === "T" && w.endsWith(i, "T"))) return "ɪ";
      return "ə";
    }
    if (name === "AH" && nxt === "M" && w.endsWith(i, "M") && w.word.endsWith("ome")) return "ʌ";
    return name === "IH" && !["o", "u", "ou"].includes(spelled) ? "ɪ" : "ə";
  }

  function unstressedIy(w, i) {
    const nxt = w.base(i + 1);
    if (nxt === null || w.endsWith(i, "Z") || w.endsWith(i, "D")) return ["i", 0];
    if (w.vowel(i + 1)) return [nxt === "IH" && w.base(i + 2) === "NG" ? "i" : "ɪ", 0];
    if (w.weakPrefix(i)) return w.nextVowelStress(i) === 1 ? ["ᵻ", 0] : ["iː", 2];
    return ["ɪ", 0];
  }

  function convert(word, arpabet, runs = null) {
    const w = new AlignedWord(word, arpabet.slice(), runs);
    const out = [];
    const emit = (phone, stress = 0) => out.push([phone, stress || 0]);
    let i = 0;
    while (i < w.count) {
      const name = w.bases[i], stress = w.stress[i] || 0;
      const nxt = w.base(i + 1);
      if (!ARPA_VOWELS.has(name)) {
        if (name === "Y" && nxt === "AH" && w.unstressed(i + 1) && ["N", "L"].includes(w.base(i - 1)) && w.base(i + 2) !== "L"
            && (!w.aligned || (w.runs.at(i) + w.runs.at(i + 1)).includes("i"))) {
          emit("iə");
          i += 2;
          continue;
        }
        emit(ARPA_CONSONANTS[name]);
        i += 1;
        continue;
      }
      const beforeVowel = w.vowel(i + 2);
      if (name === "ER") {
        emit(stress ? "ɜː" : "ɚ", stress);
        if (nxt === "R") i += 1;
        if (w.vowel(i + 1)) emit("ɹ");
      } else if (nxt === "R" && name === "AW") {
        emit("aʊ", stress);
        emit("ɚ");
        if (beforeVowel) emit("ɹ");
        i += 1;
      } else if (nxt === "R" && ["AA", "AO", "IH", "IY", "EH", "UH", "UW", "AY"].includes(name)) {
        let quality, single;
        if (name === "AA") { quality = "ɑː"; single = "ɑːɹ"; }
        else if (name === "AO") { quality = northOrForce(w, i, beforeVowel); single = quality + "ɹ"; }
        else if (name === "IH" || name === "IY") { quality = stress && ["e", "eo"].includes(w.letters(i)) ? "iə" : "ɪ"; single = "ɪɹ"; }
        else if (name === "EH") {
          const doubled = (w.runs.at(i + 1) + w.after(i + 1)).startsWith("rr") || (w.runs.at(i).endsWith("r") && w.runs.at(i + 1).startsWith("r"));
          quality = w.letters(i) === "a" && doubled ? "æ" : "ɛ";
          single = "ɛɹ";
        } else if (name === "UH" || name === "UW") { quality = single = "ʊɹ"; }
        else { quality = !w.aligned || IRE_ENDINGS.includes(w.after(i + 1)) ? "aɪɚ" : "aɪ"; single = "aɪɚ"; }
        if (beforeVowel) { emit(quality, stress); emit("ɹ"); }
        else emit(single, stress);
        i += 1;
      } else if (name === "AY" && nxt === "ER" && w.unstressed(i + 1) && w.runs.at(i + 1) === "er" && w.runs.at(i) !== "i") {
        emit("aɪ", stress);
      } else if (name === "AY" && nxt === "ER" && w.unstressed(i + 1)) {
        emit("aɪɚ", stress);
        i += 1;
        if (w.vowel(i + 1)) emit("ɹ");
      } else if (name === "AY" && nxt === "AH" && w.unstressed(i + 1)) {
        emit("aɪə", stress);
        i += 1;
      } else if (name === "IY" && nxt === "AH" && w.unstressed(i + 1) && !w.weakPrefix(i)) {
        const after = w.base(i + 2);
        if (stress !== 1 && (["L", "D", "T"].includes(after) || (after === "N" && w.base(i + 3) === "S"))) emit("ɪ");
        else { emit("iə", stress); i += 1; }
      } else if (!stress && name === "AH" && nxt === "N" && w.base(i - 1) === "T" && w.base(i + 2) !== "T"
                 && out.length >= 2 && out[out.length - 1][0] === "t" && VOWEL_SET.has(out[out.length - 2][0]) && out[out.length - 2][1]) {
        out[out.length - 1][0] = "ʔ";
        emit("n̩");
        i += 1;
      } else if (!stress && name === "AH" && nxt === "W" && w.base(i + 2) === "AH" && w.unstressed(i + 2)
                 && (w.runs.at(i) + w.runs.at(i + 1)).includes("u")) {
        emit("uː");
        i += 1;
      } else if (!stress && name === "AH" && nxt === "L" && i > 0 && !(w.vowel(i + 2) && w.stress[i + 2])) {
        if (w.letters(i) === "i" && w.vowel(i + 2)) emit("ɪ");
        else if (w.base(i - 1) === "Y" && (w.runs.at(i - 1) + w.runs.at(i)).includes("u")) emit("ʊ");
        else { emit("əl"); i += 1; }
      } else if (!stress && (name === "AH" || name === "IH")) {
        emit(reduced(w, i));
      } else if (name === "IY" && (!stress || (stress === 2 && nxt === null))) {
        emit(...unstressedIy(w, i));
      } else if (!stress && name === "AE" && i === 0 && startsWithAny(w.word, ["ad", "acc"]) && ["D", "K"].includes(nxt)) {
        emit("ɐ");
      } else if (!stress && name === "ER") {
        emit("ɚ");
      } else if (["AA", "AO"].includes(name) && w.letters(i) === "a" && endsWithAny(w.before(i), ["w", "qu"]) && nxt === "N") {
        emit("ɔ", stress);
      } else if (name === "AO" && w.letters(i) === "o") {
        emit("ɔ", stress);
      } else {
        emit(FULL_VOWELS[name], stress);
      }
      i += 1;
    }
    flap(out);
    for (let k = 0; k < out.length - 1; k++) if (out[k][0] === "n" && (out[k + 1][0] === "k" || out[k + 1][0] === "ɡ")) out[k][0] = "ŋ";
    settleStressInPlace(out);
    return out.map(([phone, stress]) => [phone, stress]);
  }

  function flap(out) {
    for (let k = 1; k < out.length - 1; k++) {
      if (out[k][0] !== "t") continue;
      const before = out[k - 1][0], [after, stress] = out[k + 1];
      if ((!VOWEL_SET.has(before) && before !== "ɹ") || before === "n̩" || before === "əl" || !VOWEL_SET.has(after) || after === "n̩") continue;
      const finalO = after === "oʊ" && (k + 2 === out.length || (k + 3 === out.length && out[k + 2][0] === "z"));
      if (stress && !finalO) continue;
      if ((after === "ə" || after === "ɚ") && k + 2 < out.length && out[k + 2][0] === "n") continue;
      out[k][0] = "ɾ";
    }
  }

  function settleStress(phones) {
    const out = phones.map(([phone, stress]) => [phone, stress]);
    settleStressInPlace(out);
    return out;
  }

  function settleStressInPlace(out) {
    const vowels = [];
    out.forEach(([phone], k) => { if (VOWEL_SET.has(phone)) vowels.push(k); });
    const primaries = vowels.filter((k) => out[k][1] === 1);
    for (const k of primaries.slice(0, -1)) out[k][1] = 2;
    vowels.forEach((k, position) => {
      if (out[k][1] !== 2) return;
      const neighbours = [position - 1, position + 1].filter((p) => p >= 0 && p < vowels.length).map((p) => vowels[p]);
      if (neighbours.some((n) => out[n][1] === 1)) out[k][1] = 0;
    });
  }

  // ---- tables.py ---------------------------------------------------------------------------------------
  const COLUMNS = R("\\t+| {2,}");
  const STRESS_CLASSES = new Set(["u", "uf", "ur", "s", "ss", "si", "pf", "p"]);
  const LINE_BREAK = /\r\n|[\n\r\v\f\x1c-\x1e\x85\u{2028}\u{2029}]/u;

  function rows(text) {
    const lines = text.split(LINE_BREAK);
    const out = [];
    lines.forEach((raw, k) => {
      const line = strip(raw);
      if (line && !line.startsWith("#")) out.push([k + 1, split(COLUMNS, line).map(strip)]);
    });
    return out;
  }

  class Tables {
    constructor(texts) { this.texts = texts; }
    rows(name) { return rows(get(this.texts, name)); }
    functionWords() {
      const table = new Map();
      for (const [number, columns] of this.rows("function_words.txt")) {
        if (![3, 4].includes(columns.length) || !STRESS_CLASSES.has(columns[1])) throw new ValueError(`function_words.txt line ${number}`);
        table.set(columns[0], [columns[1], fromText(columns[2]), columns.length === 4 ? fromText(columns[3]) : null]);
      }
      return table;
    }
    phrases() {
      const table = new Map();
      for (const [number, columns] of this.rows("phrases.txt")) {
        if (![3, 4].includes(columns.length) || !STRESS_CLASSES.has(columns[1])) throw new ValueError(`phrases.txt line ${number}`);
        const forms = columns[2].split("|").map(fromText);
        if (forms.length > 1 && (columns[1] !== "p" || forms.length !== words(columns[0]).length)) throw new ValueError(`phrases.txt line ${number}`);
        table.set(words(columns[0]).join(" "), [columns[1], forms, columns.length === 4 ? fromText(columns[3]) : null]);
      }
      return table;
    }
    pronunciations(name) {
      const table = new Map();
      for (const [where, columns] of this.rows(name)) {
        if (columns.length !== 2) throw new ValueError(`${where}: expected a word and its pronunciation`);
        if (words(columns[0]).length === 1) table.set(lower(columns[0]), pronunciation(columns[1]));
      }
      return table;
    }
    names(name) {
      const table = new Map();
      for (const [where, columns] of this.rows(name)) {
        const ws = words(lower(columns[0]));
        if (columns.length !== 2 || ws.length < 2) continue;
        const parts = columns[1].split("|").map(strip);
        if (parts.length !== ws.length) throw new ValueError(`${where}: one pronunciation per word`);
        table.set(ws.join(" "), parts.map(pronunciation));
      }
      return table;
    }
    mapping(name) {
      const table = {};
      for (const [, columns] of this.rows(name)) table[columns[0]] = columns.length > 1 ? columns[1] : "";
      return table;
    }
  }
  function pronunciation(text) {
    const symbols = words(text);
    return isArpabet(symbols) ? ["arpabet", symbols] : ["phones", fromText(text)];
  }

  // ---- morphology.py -----------------------------------------------------------------------------------
  const SHORT_STEM = rx("(?:^|[^aeiou])[aeiou](?:[bcdfgklmnprstvz]|th)$");
  const M_SIBILANT = new Set(["s", "z", "ʃ", "ʒ", "tʃ", "dʒ"]);
  const M_VOICELESS = new Set(["p", "t", "k", "f", "θ", "ʔ"]);
  const M_SPLIT = { "ɑːɹ": "ɑː", "ɔːɹ": "ɔː", "oːɹ": "oː", "ɪɹ": "ɪ", "ɛɹ": "ɛ" };
  const M_RHOTIC = new Set(["ɚ", "ɜː", "aɪɚ", "ʊɹ"]);
  const CONSONANT_LETTERS = new Set("bcdfghjklmnpqrstvwxz");
  const P = (...phones) => phones.map((phone) => [phone, 0]);
  const ENDINGS = [
    ["fully", P("f", "əl", "i")], ["lessly", P("l", "ə", "s", "l", "i")], ["ness", P("n", "ə", "s")],
    ["less", P("l", "ə", "s")], ["ments", P("m", "ə", "n", "t", "s")], ["ment", P("m", "ə", "n", "t")],
    ["ful", P("f", "əl")], ["ship", [["ʃ", 0], ["ɪ", 2], ["p", 0]]], ["hood", [["h", 0], ["ʊ", 2], ["d", 0]]],
    ["like", [["l", 0], ["aɪ", 2], ["k", 0]]], ["dom", P("d", "ə", "m")], ["ish", P("ɪ", "ʃ")],
    ["able", P("ə", "b", "əl")], ["ism", P("ɪ", "z", "ə", "m")], ["ist", P("ɪ", "s", "t")],
    ["wards", P("w", "ɚ", "d", "z")], ["ward", P("w", "ɚ", "d")], ["age", P("ɪ", "dʒ")],
    ["ers", P("ɚ", "z")], ["er", P("ɚ")], ["est", P("ɪ", "s", "t")], ["y", P("i")],
  ];
  const twice = (x) => [x, x];
  const M_PREFIXES = [
    ["under", twice([["ʌ", 2], ["n", 0], ["d", 0], ["ɚ", 0]])], ["over", twice([["oʊ", 2], ["v", 0], ["ɚ", 0]])],
    ["anti", twice([["æ", 2], ["n", 0], ["t", 0], ["i", 0]])], ["semi", twice([["s", 0], ["ɛ", 2], ["m", 0], ["i", 0]])],
    ["non", twice([["n", 0], ["ɑː", 2], ["n", 0]])], ["out", twice([["aʊ", 2], ["t", 0]])],
    ["dis", [P("d", "ɪ", "s"), [["d", 0], ["ɪ", 2], ["s", 0]]]], ["mis", [P("m", "ɪ", "s"), [["m", 0], ["ɪ", 2], ["s", 0]]]],
    ["un", [P("ʌ", "n"), [["ʌ", 2], ["n", 0]]]], ["in", [P("ɪ", "n"), [["ɪ", 2], ["n", 0]]]],
    ["re", [P("ɹ", "ᵻ"), [["ɹ", 0], ["iː", 2]]]], ["pre", twice([["p", 0], ["ɹ", 0], ["iː", 2]])],
  ];
  const BRITISH = [["isation", "ization"], ["ising", "izing"], ["ised", "ized"], ["ises", "izes"], ["ise", "ize"], ["yse", "yze"],
    ["our", "or"], ["ourable", "orable"], ["ours", "ors"], ["oured", "ored"], ["ouring", "oring"],
    ["tre", "ter"], ["tres", "ters"], ["bre", "ber"], ["ence", "ense"], ["ogue", "og"], ["mme", "m"],
    ["lled", "led"], ["lling", "ling"], ["ller", "ler"], ["oul", "ol"], ["ae", "e"], ["enqui", "inqui"]];
  const truthy = (x) => x !== null && x !== undefined && (!Array.isArray(x) || x.length > 0);

  function join(stem, ending) {
    stem = stem.map((p) => p.slice());
    if (!stem.length || !ending.length) return stem.concat(ending.map((p) => p.slice()));
    const [last, stress] = stem[stem.length - 1], [first, firstStress] = ending[0];
    if (VOWEL_SET.has(first)) {
      if (has(M_SPLIT, last)) stem.splice(stem.length - 1, 1, [M_SPLIT[last], stress], ["ɹ", 0]);
      else if (M_RHOTIC.has(last)) stem.push(["ɹ", 0]);
      else if (last === "t" && stem.length > 1 && (VOWEL_SET.has(stem[stem.length - 2][0]) || stem[stem.length - 2][0] === "ɹ") && !firstStress
               && !(first === "ə" && ending.length > 1 && ending[1][0] === "n")) stem[stem.length - 1] = ["ɾ", 0];
      else if (last === "i") stem[stem.length - 1] = first !== "ɪ" ? ["ɪ", stress] : ["i", stress];
    } else if (last === "i" && ["n", "l", "f", "m"].includes(first)) {
      stem[stem.length - 1] = ["ɪ", stress];
    } else if (last === "n" && (first === "k" || first === "ɡ")) {
      stem[stem.length - 1] = ["ŋ", 0];
    }
    return stem.concat(ending.map((p) => p.slice()));
  }
  function plural(stem) {
    const last = stem.at(-1)[0];
    return join(stem, M_SIBILANT.has(last) ? P("ᵻ", "z") : M_VOICELESS.has(last) ? P("s") : P("z"));
  }
  function past(stem) {
    const last = stem.at(-1)[0];
    if (["t", "d", "ɾ"].includes(last)) return join(stem, P("ᵻ", "d"));
    return join(stem, M_VOICELESS.has(last) || ["s", "ʃ", "tʃ"].includes(last) ? P("t") : P("d"));
  }
  function adverb(stem) {
    const last = stem.at(-1)[0];
    return join(stem, last === "əl" || last === "l" ? P("i") : P("l", "i"));
  }
  function stems(body) {
    const out = [body];
    if (body.length > 2 && body.at(-1) === body.at(-2) && CONSONANT_LETTERS.has(body.at(-1))) out.push(body.slice(0, -1));
    if (body && (CONSONANT_LETTERS.has(body.at(-1)) || body.at(-1) === "u")) {
      const single = search(SHORT_STEM, body) !== null;
      out.splice(single ? 0 : out.length, 0, body + "e");
    }
    if (body.endsWith("i") && body.length > 2) out.push(body.slice(0, -1) + "y");
    return out;
  }
  const secondary = (phones) => phones.map(([phone, stress]) => [phone, stress === 1 ? 2 : stress]);
  const compound = (first, second) => join(first, secondary(second));

  function derive(word, lookup, part = null, depth = 0) {
    part = part || lookup;
    const stem = (spelling) => {
      if (spelling.length < 2) return null;
      const found = lookup(spelling);
      if (truthy(found)) return found;
      return depth < 2 ? derive(spelling, lookup, part, depth + 1) : null;
    };
    const first = (candidates, build) => {
      for (const spelling of candidates) {
        const found = stem(spelling);
        if (truthy(found)) return build(found);
      }
      return null;
    };
    let found;
    if (word.endsWith("'s")) return first([word.slice(0, -2)], plural);
    if (word.endsWith("s'")) return first([word.slice(0, -1)], (f) => f);
    if (word.endsWith("ies") && word.length > 4) { found = first([word.slice(0, -3) + "y"], plural); if (truthy(found)) return found; }
    if (word.endsWith("ves") && word.length > 4) {
      found = first([word.slice(0, -3) + "f", word.slice(0, -3) + "fe"], (f) => (f.at(-1)[0] === "f" ? join(f.slice(0, -1).concat([["v", 0]]), P("z")) : null));
      if (truthy(found)) return found;
    }
    if (word.endsWith("es") && word.length > 3) { found = first("sxzh".includes(word.at(-3)) ? [word.slice(0, -2)] : [], plural); if (truthy(found)) return found; }
    if (word.endsWith("s") && !endsWithAny(word, ["ss", "us", "is"]) && word.length > 3) { found = first([word.slice(0, -1)], plural); if (truthy(found)) return found; }
    if (word.endsWith("ied") && word.length > 4) { found = first([word.slice(0, -3) + "y"], past); if (truthy(found)) return found; }
    if (word.endsWith("ed") && word.length > 3) { found = first(stems(word.slice(0, -2)), past); if (truthy(found)) return found; }
    if (word.endsWith("ing") && word.length > 4) { found = first(stems(word.slice(0, -3)), (f) => join(f, P("ɪ", "ŋ"))); if (truthy(found)) return found; }
    if (word.endsWith("ily") && word.length > 4) { found = first([word.slice(0, -3) + "y"], adverb); if (truthy(found)) return found; }
    if (word.endsWith("ly") && word.length > 3) {
      found = first(word.endsWith("bly") || word.endsWith("ply") ? [word.slice(0, -2), word.slice(0, -2) + "le", word.slice(0, -1) + "e"] : [word.slice(0, -2)], adverb);
      if (truthy(found)) return found;
    }
    for (const [spelling, phones] of ENDINGS) {
      if (word.endsWith(spelling) && word.length - spelling.length > 2) {
        const body = word.slice(0, -spelling.length);
        if (spelling === "y" && !CONSONANT_LETTERS.has(body.at(-1))) continue;
        const vowelFirst = VOWEL_SET.has(phones[0][0]);
        found = first(vowelFirst ? stems(body) : body.endsWith("i") ? [body.slice(0, -1) + "y", body] : [body], (f) => join(f, phones));
        if (truthy(found)) return found;
      }
    }
    for (const [spelling, [plain, stressed]] of M_PREFIXES) {
      if (word.startsWith(spelling) && word.length - spelling.length > 2) {
        found = stem(word.slice(spelling.length));
        if (truthy(found)) {
          const vowels = found.filter(([phone]) => VOWEL_SET.has(phone)).map(([, stress]) => stress);
          const head = vowels.length && vowels[0] === 1 ? plain : stressed;
          return join(head, found);
        }
      }
    }
    for (const [old, replacement] of BRITISH) {
      if (word.includes(old)) {
        found = stem(word.replace(old, replacement));
        if (truthy(found)) return found;
      }
    }
    if (depth === 0) {
      let best = null;
      for (let cut = 4; cut < word.length - 3; cut++) {
        const left = part(word.slice(0, cut)), right = part(word.slice(cut));
        if (truthy(left) && truthy(right)) {
          const score = Math.min(cut, word.length - cut);
          if (best === null || score > best[0]) best = [score, left, right];
        }
      }
      if (best) return compound(best[1], best[2]);
    }
    return null;
  }

  // ---- sentence.py -------------------------------------------------------------------------------------
  const S_SPLIT = { "ɑːɹ": "ɑː", "ɔːɹ": "ɔː", "oːɹ": "oː", "ɪɹ": "ɪ", "ɛɹ": "ɛ", "ʊɹ": "ʊ" };
  const S_RHOTIC = new Set(["ɚ", "ɜː", "aɪɚ"]);
  const NO_LINK_BEFORE = new Set(["and", "or", "if"]);
  const FLAP_BEFORE_VOWEL = new Set(["it", "at"]);

  class Unit {
    constructor(word, phones, kind = "c", final = null, attach = false, source = "") {
      Object.assign(this, { word, phones, kind, final, attach, source });
    }
    get lastWord() { return rpartition(this.word, " ")[2]; }
    get firstWord() { return partition(this.word, " ")[0]; }
  }
  const unstress = (phones) => phones.map(([phone]) => [phone, 0]);

  function promote(phones) {
    const marks = phones.map(([, stress]) => stress);
    if (marks.includes(1)) return phones.map((p) => p.slice());
    let target;
    if (marks.includes(2)) target = marks.lastIndexOf(2);
    else { target = phones.findIndex(([phone]) => VOWEL_SET.has(phone)); if (target < 0) target = null; }
    return phones.map(([phone, stress], k) => [phone, k === target ? 1 : stress]);
  }

  function stressClause(units) {
    let primaryFollows = false;
    for (let index = units.length - 1; index >= 0; index--) {
      const unit = units[index];
      const promoted = () => (unit.final !== null ? unit.final : promote(unit.phones));
      let phones;
      if (unit.kind === "u") phones = unstress(unit.phones);
      else if (unit.kind === "uf") phones = index === units.length - 1 ? promoted() : unstress(unit.phones);
      else if (unit.kind === "ur") phones = primaryFollows ? unstress(unit.phones) : promoted();
      else if (unit.kind === "s") phones = primaryFollows ? unit.phones : promoted();
      else if (unit.kind === "si") phones = !primaryFollows ? promoted() : index === 0 ? unit.phones : unstress(unit.phones);
      else if (unit.kind === "pf") phones = index === units.length - 1 ? promoted() : unit.phones;
      else phones = unit.phones;
      unit.phones = phones.map((p) => p.slice());
      primaryFollows = primaryFollows || unit.phones.some(([, stress]) => stress === 1);
    }
    if (units.length && !units.some((unit) => unit.phones.some(([, stress]) => stress))) {
      for (let k = units.length - 1; k >= 0; k--) {
        if (units[k].phones.some(([phone]) => VOWEL_SET.has(phone))) { units[k].phones = promote(units[k].phones); break; }
      }
    }
  }

  function joinClause(units) {
    for (let k = 0; k + 1 < units.length; k++) {
      const unit = units[k], following = units[k + 1];
      if (!unit.phones.length || !following.phones.length || !VOWEL_SET.has(following.phones[0][0])) continue;
      const [last, stress] = unit.phones.at(-1);
      if (unit.lastWord === "the" && last === "ə") unit.phones = unit.phones.slice(0, -1).concat([["ɪ", stress]]);
      else if (unit.lastWord === "to" && last === "ə") unit.phones = unit.phones.slice(0, -1).concat([["ʊ", stress]]);
      else if (NO_LINK_BEFORE.has(following.firstWord)) continue;
      else if (has(S_SPLIT, last)) unit.phones = unit.phones.slice(0, -1).concat([[S_SPLIT[last], stress], ["ɹ", 0]]);
      else if (S_RHOTIC.has(last)) unit.phones = unit.phones.concat([["ɹ", 0]]);
      else if (last === "t" && FLAP_BEFORE_VOWEL.has(unit.word) && unit.phones.length > 1 && VOWEL_SET.has(unit.phones.at(-2)[0])) {
        unit.phones = unit.phones.slice(0, -1).concat([["ɾ", 0]]);
      }
    }
  }

  function groups(units) {
    const out = [];
    for (const unit of units) {
      if (unit.attach && out.length) out[out.length - 1] = out[out.length - 1].concat(unit.phones);
      else out.push(unit.phones.slice());
    }
    return out.filter((group) => group.length);
  }

  // ---- homographs.py -----------------------------------------------------------------------------------
  const H_CLASSES = [
    ["TO", "to"],
    ["DET", "the a an this that these those every each no some any another such what which whose"],
    ["POSS", "my your his her its our their"],
    ["SUBJ", "i you he she it we they who"],
    ["OBJ", "me him us them"],
    ["MODAL", "will would shall should can could may might must 'll 'd won't can't cannot couldn't shouldn't wouldn't let's please"],
    ["DO", "do does did don't doesn't didn't"],
    ["BE", "am is are was were be been being isn't aren't wasn't weren't i'm you're we're they're he's she's it's that's there's"],
    ["HAVE", "have has had having haven't hasn't hadn't i've you've we've they've"],
    ["NEG", "not never"],
    ["PREP", "of in on at by for from with into onto over under about after before between through during without within " +
             "against among around across behind beside near upon toward towards as than like"],
    ["CONJ", "and or but nor so yet if because although though while when where unless until since whether"],
    ["ADV", "very too quite rather more most less least also just only even still always often sometimes usually again " +
            "now then here there soon already almost"],
  ];
  const H_TAG = new Map();
  for (const [name, list] of H_CLASSES) for (const w of list.split(" ")) H_TAG.set(w, name);
  const SENTENCE_END = new Set([".", "?", "!"]);
  const WINDOW = 6;
  const firstCp = (s) => cpAt(s, 0);

  function tag(token) {
    if (token === "<s>" || token === "</s>") return token;
    const head = firstCp(token);
    if (!isalpha(head)) return isdigit(head) ? "NUM" : "PUNCT";
    const low = lower(token);
    if (H_TAG.has(low)) return H_TAG.get(low);
    for (const [ending, name] of [["'s", "POS"], ["ly", "LY"], ["ing", "ING"], ["ed", "ED"]]) {
      if (low.endsWith(ending) && plen(low) > ending.length + 1) return name;
    }
    if (low.endsWith("s") && !low.endsWith("ss") && plen(low) > 3) return "S";
    return isupper(head) ? "CAP" : "W";
  }

  function features(tokens, index) {
    const at = (offset) => {
      const position = index + offset;
      return position >= 0 && position < tokens.length ? tokens[position] : offset < 0 ? "<s>" : "</s>";
    };
    const p2 = at(-2), p1 = at(-1), n1 = at(1), n2 = at(2);
    const tp2 = tag(p2), tp1 = tag(p1), tn1 = tag(n1), tn2 = tag(n2);
    const out = ["tp1=" + tp1, "tp2=" + tp2, "tn1=" + tn1, "tn2=" + tn2, "tp2p1=" + tp2 + "+" + tp1, "tp1n1=" + tp1 + "+" + tn1,
      "tn1n2=" + tn1 + "+" + tn2];
    if (index === 0) out.push("first");
    else if (isupper(firstCp(tokens[index]))) out.push("cap");
    out.push("p1=" + lower(p1), "p2=" + lower(p2), "n1=" + lower(n1), "n2=" + lower(n2));
    const target = lower(tokens[index]);
    const nearby = new Set();
    for (const t of tokens.slice(Math.max(0, index - WINDOW), index + WINDOW + 1)) if (isalpha(firstCp(t))) nearby.add(lower(t));
    nearby.delete(target);
    for (const word of [...nearby].sort(cmp)) out.push("w=" + word);
    return out;
  }
  const shared = (names) => names.filter((name) => name[0] === "t" || name === "first" || name === "cap");

  function sentenceOf(tokens, index) {
    let start = index;
    while (start > 0 && !SENTENCE_END.has(tokens[start - 1])) start--;
    let end = index;
    while (end < tokens.length && !SENTENCE_END.has(tokens[end])) end++;
    return [tokens.slice(start, end + 1), index - start];
  }

  const SKIP_FOR_BEFORE = new Set(["ADV", "NEG", "LY"]);
  const SKIP_FOR_AUX = new Set([...SKIP_FOR_BEFORE, "SUBJ", "OBJ", "DET", "POSS"]);
  function left(tokens, index, skip) {
    for (let position = index - 1; position > Math.max(index - 6, -1); position--) {
      const token = tokens[position];
      if (skip.has(tag(token))) continue;
      const low = lower(token);
      return low.endsWith("'ll") ? "will" : low.endsWith("'ve") ? "have" : low;
    }
    return "<s>";
  }

  function readRules(text) {
    const rules = new Map();
    for (const [number, columns] of rows(text)) {
      if (columns.length !== 3) throw new ValueError(`homograph_rules.txt line ${number}`);
      const conditions = [];
      for (const part of columns[2].split(";")) {
        const [name, , list] = partition(strip(part), ":");
        if (!["aux", "before", "next", "near", "first", "joined"].includes(name) || (name === "first") === (strip(list) !== "")) {
          throw new ValueError(`homograph_rules.txt line ${number}: bad condition`);
        }
        conditions.push([name, new Set(words(list))]);
      }
      if (!rules.has(columns[0])) rules.set(columns[0], []);
      rules.get(columns[0]).push([columns[1], conditions]);
    }
    return rules;
  }

  function holds(condition, list, tokens, index, joined) {
    if (condition === "first") return index === 0;
    if (condition === "joined") return Boolean(joined) && index > 0 && list.has(lower(tokens[index - 1]));
    if (condition === "aux") return list.has(left(tokens, index, SKIP_FOR_AUX));
    if (condition === "before") return list.has(left(tokens, index, SKIP_FOR_BEFORE));
    if (condition === "next") return list.has(index + 1 < tokens.length ? lower(tokens[index + 1]) : "</s>");
    return tokens.slice(Math.max(0, index - WINDOW), index + WINDOW + 1).some((token) => list.has(lower(token)));
  }

  class Homographs {
    constructor(data, rulesText) {
      this.words = data.words;
      this.pooled = data.pooled;
      this.pluralBases = new Set(data.plural_bases || []);
      this.rules = readRules(rulesText);
    }
    covers(word) { return has(this.words, word) || (word.endsWith("s") && this.pluralBases.has(word.slice(0, -1))); }
    pronounce(word, tokens, index, joined = false) {
      if (has(this.words, word)) return [this.choose(word, tokens, index, joined).phones, false];
      const entry = get(this.words, word.slice(0, -1));
      const adjective = entry.senses.find((sense) => sense.role === "n") || null;
      if (adjective && adjective.id === "adj") return [entry.senses.find((sense) => sense.role === "v").phones, true];
      return [this.choose(word.slice(0, -1), tokens, index, joined).phones, true];
    }
    scores(word, tokens, index) {
      const entry = get(this.words, word);
      const names = features(...sentenceOf(tokens, index));
      const model = entry.model;
      const weight = (table, name) => (has(table, name) ? table[name] : 0.0);
      if (model.kind === "noun-verb") {
        let verb = this.pooled.bias + pysum(shared(names).map((name) => weight(this.pooled.weights, name)));
        verb += model.bias + pysum(names.map((name) => weight(model.weights, name)));
        return entry.senses.map((sense) => (sense.role === "v" ? verb : -verb));
      }
      return model.bias.map((bias, k) => bias + pysum(names.map((name) => weight(model.weights[k], name))));
    }
    choose(word, tokens, index, joined = false) {
      const entry = get(this.words, word);
      const [sentence, position] = sentenceOf(tokens, index);
      for (const [sense, conditions] of this.rules.get(word) || []) {
        if (conditions.every(([name, list]) => holds(name, list, sentence, position, joined))) {
          return entry.senses.find((s) => s.id === sense);
        }
      }
      const scores = this.scores(word, tokens, index);
      let best = 0;
      for (let k = 1; k < scores.length; k++) if (scores[k] > scores[best]) best = k;
      return entry.senses[best];
    }
  }

  // ---- g2p.py ------------------------------------------------------------------------------------------
  // The arithmetic of NumPy in float32: products summed in float64 then rounded (BLAS sums in float32 with its
  // own order, so a value can differ in its last bit), every elementwise step rounded to float32.
  const f32 = Math.fround;

  class Model {
    constructor(settings, weights) {
      this.settings = settings;
      this.w = weights;                              // name -> {data: Float32Array, shape}
      this.letters = new Map([...settings.letters].map((letter, k) => [letter, k + 1]));
      this.symbols = settings.symbols;
    }
    spellable(word) {
      const plain = lower(word).normalize("NFKD");
      return stripChars([...plain].filter((c) => this.letters.has(c)).join(""), "'");
    }
    // x (frames x n) @ weight.T (out x n) + bias -> frames x out, float32
    static linear(x, frames, n, weight, bias, out) {
      const y = new Float32Array(frames * out);
      for (let t = 0; t < frames; t++) {
        const row = t * n;
        for (let o = 0; o < out; o++) y[t * out + o] = f32(f32(dot(x, row, weight, o * n, n)) + bias[o]);
      }
      return y;
    }
    gru(x, frames, n, layer, suffix, reverse) {
      const w = this.w;
      const wih = w[`gru__weight_ih_l${layer}${suffix}`], whh = w[`gru__weight_hh_l${layer}${suffix}`];
      const bih = w[`gru__bias_ih_l${layer}${suffix}`].data, bhh = w[`gru__bias_hh_l${layer}${suffix}`].data;
      const size = whh.shape[1], gates = 3 * size;
      const inputs = Model.linear(x, frames, n, wih.data, bih, gates);
      const hh = whh.data;
      let state = new Float32Array(size), next = new Float32Array(size);
      const out = new Float32Array(frames * size);
      const recurrent = new Float32Array(gates);
      for (let step = 0; step < frames; step++) {
        const t = reverse ? frames - 1 - step : step;
        for (let o = 0; o < gates; o++) recurrent[o] = f32(f32(dot(hh, o * size, state, 0, size)) + bhh[o]);
        const row = t * gates;
        for (let k = 0; k < size; k++) {
          const reset = sigmoid(f32(inputs[row + k] + recurrent[k]));
          const update = sigmoid(f32(inputs[row + size + k] + recurrent[size + k]));
          const candidate = f32(Math.tanh(f32(inputs[row + 2 * size + k] + f32(reset * recurrent[2 * size + k]))));
          next[k] = f32(f32(f32(1.0 - update) * candidate) + f32(update * state[k]));
        }
        [state, next] = [next, state];
        out.set(state, t * size);
      }
      return { out, size };
    }
    logProbabilities(word) {
      const w = this.w, slots = this.settings.slots;
      const ids = [...word].map((c) => this.letters.get(c));
      const width = w.embed__weight.shape[1];
      let frames = ids.length * slots, n = width;
      let x = new Float32Array(frames * width);
      ids.forEach((id, k) => {
        for (let s = 0; s < slots; s++) {
          for (let d = 0; d < width; d++) x[(k * slots + s) * width + d] = f32(w.embed__weight.data[id * width + d] + w.slot.data[s * width + d]);
        }
      });
      if (this.settings.convs) throw new Error("convolution blocks are not ported: the shipped model has none");
      for (let layer = 0; layer < this.settings.layers; layer++) {
        const a = this.gru(x, frames, n, layer, "", false), b = this.gru(x, frames, n, layer, "_reverse", true);
        const size = a.size;
        const both = new Float32Array(frames * 2 * size);
        for (let t = 0; t < frames; t++) {
          both.set(a.out.subarray(t * size, (t + 1) * size), t * 2 * size);
          both.set(b.out.subarray(t * size, (t + 1) * size), t * 2 * size + size);
        }
        x = both;
        n = 2 * size;
      }
      const classes = w.out__weight.shape[0];
      const logits = Model.linear(x, frames, n, w.out__weight.data, w.out__bias.data, classes);
      // log softmax, in float32: the shift, exp, a pairwise sum (NumPy's order), log
      const out = [];
      for (let t = 0; t < frames; t++) {
        const row = logits.subarray(t * classes, (t + 1) * classes);
        let max = -Infinity;
        for (const v of row) if (v > max) max = v;
        const shifted = Array.from(row, (v) => f32(v - max));
        const total = pairwiseSum(shifted.map((v) => f32(Math.exp(v))));
        const lse = f32(Math.log(total));
        out.push(shifted.map((v) => f32(v - lse)));
      }
      return out;
    }
    predict(word) {
      word = this.spellable(word);
      if (!word) return [];
      const out = [];
      let previous = 0;
      for (const row of this.logProbabilities(word)) {
        let label = 0;
        for (let k = 1; k < row.length; k++) if (row[k] > row[label]) label = k;
        if (label !== previous && label !== 0) out.push(this.symbols[label - 1]);
        previous = label;
      }
      return out;
    }
  }
  function sigmoid(x) { return f32(1.0 / f32(1.0 + f32(Math.exp(-x)))); }
  // a[i..i+n) . b[j..j+n) in float64 (each float32 product is exact in float64)
  function dot(a, i, b, j, n) {
    let s0 = 0, s1 = 0, s2 = 0, s3 = 0, k = 0;
    for (; k + 3 < n; k += 4) {
      s0 += a[i + k] * b[j + k];
      s1 += a[i + k + 1] * b[j + k + 1];
      s2 += a[i + k + 2] * b[j + k + 2];
      s3 += a[i + k + 3] * b[j + k + 3];
    }
    for (; k < n; k++) s0 += a[i + k] * b[j + k];
    return (s0 + s1) + (s2 + s3);
  }
  // NumPy's pairwise summation for float32 (blocks of 8 partial sums below 128 elements).
  function pairwiseSum(a, lo = 0, n = a.length) {
    if (n < 8) {
      let res = -0.0;
      for (let k = 0; k < n; k++) res = f32(res + a[lo + k]);
      return res;
    }
    if (n <= 128) {
      const r = [];
      for (let k = 0; k < 8; k++) r.push(a[lo + k]);
      let k = 8;
      for (; k < n - (n % 8); k += 8) for (let j = 0; j < 8; j++) r[j] = f32(r[j] + a[lo + k + j]);
      let res = f32(f32(f32(r[0] + r[1]) + f32(r[2] + r[3])) + f32(f32(r[4] + r[5]) + f32(r[6] + r[7])));
      for (; k < n; k++) res = f32(res + a[lo + k]);
      return res;
    }
    let n2 = Math.floor(n / 2);
    n2 -= n2 % 8;
    return f32(pairwiseSum(a, lo, n2) + pairwiseSum(a, lo + n2, n - n2));
  }

  // ---- normalize.py ------------------------------------------------------------------------------------
  const SPELL = "\u{e000}", END = "\u{e001}", INLINE = "\u{e003}", INLINE_BASE = 0xe100;
  const IS = "\u{e006}", ANGLE_MARK = "\u{e005}", FORMULA_SIGN = "\u{e004}";
  const ONES = ("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen " +
    "seventeen eighteen nineteen").split(" ");
  const TENS = "twenty thirty forty fifty sixty seventy eighty ninety".split(" ");
  const SCALES = ["thousand", "million", "billion", "trillion", "quadrillion", "quintillion"];
  const ORDINAL = { one: "first", two: "second", three: "third", five: "fifth", eight: "eighth", nine: "ninth", twelve: "twelfth" };
  const MONTHS = "January February March April May June July August September October November December".split(" ");
  const MONTH_NUMBER = {};
  MONTHS.forEach((name, k) => { MONTH_NUMBER[name.toLowerCase().slice(0, 3)] = k + 1; });
  const MONTH_PATTERN = MONTHS.join("|") + "|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sept|Sep|Oct|Nov|Dec";
  const DAY_NAMES = { mon: "Monday", tue: "Tuesday", wed: "Wednesday", thu: "Thursday", fri: "Friday", sat: "Saturday", sun: "Sunday" };
  const DAY_PATTERN = "Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday|Thurs|Thur|Tues|Mon|Tue|Wed|Thu|Fri|Sat|Sun";
  const CURRENCY = { "$": ["dollar", "dollars", "cent", "cents"], "£": ["pound", "pounds", "penny", "pence"],
    "€": ["euro", "euros", "cent", "cents"], "¥": ["yen", "yen", "", ""], "₹": ["rupee", "rupees", "paisa", "paise"] };
  Object.assign(CURRENCY, { USD: CURRENCY["$"], EUR: CURRENCY["€"], GBP: CURRENCY["£"], JPY: CURRENCY["¥"],
    INR: CURRENCY["₹"], CHF: ["Swiss franc", "Swiss francs", "centime", "centimes"],
    CAD: ["Canadian dollar", "Canadian dollars", "cent", "cents"],
    AUD: ["Australian dollar", "Australian dollars", "cent", "cents"], CNY: ["yuan", "yuan", "", ""] });
  const CURRENCY_CODES = "USD|EUR|GBP|JPY|INR|CHF|CAD|AUD|CNY";
  const SCALE_LETTER = { k: "thousand", m: "million", b: "billion", bn: "billion", t: "trillion" };
  const FRACTION = { 2: ["half", "halves"], 4: ["quarter", "quarters"] };
  const SYMBOL_WORDS = { "&": "and", "@": "at", "+": "plus", "=": "equals", "%": "percent", "×": "times", "÷": "divided by",
    "±": "plus or minus", "©": "copyright", "§": "section", "°": "degrees",
    "≥": "greater than or equal to", "≤": "less than or equal to", "≠": "not equal to", "≈": "approximately",
    "∞": "infinity", "√": "square root of", "π": "pi" };
  const COMPARISONS = { ">=": "greater than or equal to", "<=": "less than or equal to", "!=": "not equal to" };
  const INITIALISMS = new Set("US IT AI ID UN EU UK LA AD BC AM PM OS PE PA MA LED SAT RIP ER".split(" "));
  const SOUNDS = new Set("sh shh ss zz hm hmm mm tsk psst pfft hmph psh pssh".split(" "));
  const COMPARED = RD(String.raw`(?P<gap>[ \t]?)(?P<sign>>=|<=|!=|[<>\u{2265}\u{2264}\u{2260}])(?P=gap)`);
  const VALUE_BEFORE = rx(String.raw`(?:(?<![\w.,])\d[\d.,]*%?|(?<![\w'\u{2019}.-])[A-Za-z]|[)\]])$`);
  const VALUE_AFTER = RM(String.raw`(?:[-+]?[$\u{a3}\u{20ac}\u{a5}\u{20b9}]?\d[\d.,]*%?(?!\w)|\.\d|[A-Za-z](?![\w'\u{2019}])|[(\[])`);
  const CLAUSE_OPENERS = new Set(("if elif while when whenever where whether that because since as and or but so then think thinks " +
    "thought know knows say says said suppose assume means implies hence thus therefore although though " +
    "unless until once else also clearly obviously indeed here now again first next finally similarly " +
    "otherwise however moreover").split(" "));
  const SIGN = String.raw`[<>+*/^\u{2264}\u{2265}\u{2260}\u{2248}\u{d7}\u{f7}\u{b1}]`;
  const SPACED_SIGN = String.raw` [<>+*\u{2264}\u{2265}\u{2260}\u{2248}\u{d7}\u{f7}\u{b1}] `;
  const NO_PHRASE = String.raw`(?![\w'\u{2019}-]|\s+(?!(?:and|or|then|if|when|where|while|for|is|are|was|but|so|in|to)\b)[A-Za-z])`;
  const LETTER_PLACES = [
    String.raw`(?<![\w'\u{2019}.-])(?P<letter>[Aakur])(?=` + IS + String.raw`?\s?[<>!=]?=)`,
    String.raw`[<>!=]?=\s?(?P<letter>[Aakur])` + NO_PHRASE,
    String.raw`(?<![\w'\u{2019}.-])(?P<letter>[Aakur])(?=` + IS + "?" + SIGN + String.raw`(?:[\w($]|\.\d))`,
    String.raw`(?<=[\w)` + IS + "]" + SIGN + String.raw`)(?<![Ww]/)(?P<letter>[Aakur])(?![\w'\u{2019}-])`,
    String.raw`(?<![\w'\u{2019}.-])(?P<letter>[Aakur])(?=` + IS + "?" + SPACED_SIGN + String.raw`(?:[A-Za-z]|\d[\d.,]*)(?![\w'\u{2019}]))`,
    String.raw`(?<![\w'\u{2019}.])(?:[A-Za-z]|\d[\d.,]*)` + IS + "?" + SPACED_SIGN + String.raw`(?P<letter>[Aakur])` + NO_PHRASE,
    String.raw`(?<=\()(?P<letter>[Aakur])(?=\)|, )`,
    String.raw`(?<=, )(?P<letter>[Aakur])(?=\))`,
    String.raw`(?:(?<=\s)|^)(?P<letter>[akur])(?=\)\s)`,
  ].map((p) => RD(p));
  const ANGLES = R(String.raw`(?<![\w)` + END + IS + String.raw`])<[A-Za-z0-9/!?][^<>(){}+*;|\n]*>` +
    String.raw`|(?<=[\w` + END + IS + String.raw`])<[^<>(){}+*;|\n]{1,80}>(?![\w$])` +
    String.raw`|</?[A-Za-z][\w-]+/?>`);

  function withoutAngles(text) {
    while (text.includes("<") && text.includes(">")) {
      const marked = sub(ANGLES, (m) => ANGLE_MARK + m.group(0).slice(1, -1) + ANGLE_MARK, text);
      if (marked === text) break;
      text = marked;
    }
    return text;
  }

  const JOINED_COMPARISON = R(String.raw`(?<=[\w)` + END + IS + String.raw`])[<>](?=[\w(` + SPELL + String.raw`]|\.\d| minus \d)`);
  const FIGURE_BEFORE = rx(String.raw`(?<![\w.,])\d[\d.,]*%?$`);
  const FIGURE_AFTER = RM(String.raw`(?:\d[\d.,]*%?(?!\w)|\.\d| minus \d)`);
  const LETTER_BEFORE = rx(String.raw`(?:(?<![\w'\u{2019}.])[A-Za-z]|[)` + END + "])$");
  const LETTER_AFTER = RM(String.raw`(?:[A-Za-z](?![\w'\u{2019}])|[(` + SPELL + "])");
  const SENTENCE_STARTERS = new Set(("the he she it i we they this that there but and in on at a an if when what why how who my our " +
    "you then so now after before however").split(" "));
  const YEAR_CONTEXT = new Set(("in since by from until till before after during around circa of year early late mid summer winter " +
    "spring fall autumn class").split(" "));
  const CARDINAL_AFTER = new Set(("war chapter part book volume act scene section phase type class level stage title article " +
    "appendix world").split(" "));
  const LABELLED = new Set(("plan vitamin grade type option class section part team group exhibit appendix point model series " +
    "hepatitis level").split(" "));
  const ROMAN = { II: 2, III: 3, IV: 4, VI: 6, VII: 7, VIII: 8, IX: 9, XI: 11, XII: 12, XIII: 13, XIV: 14, XV: 15, XVI: 16, XVIII: 18, XX: 20 };
  const LEADING_APOSTROPHE = new Set(["'em", "'tis", "'twas", "'til", "'cause", "'bout", "'round", "'n"]);
  const EMOJI = R(String.raw`[\u{1F000}-\u{1FAFF}\u{2600}-\u{27BF}\u{1F1E6}-\u{1F1FF}\u{2190}-\u{21FF}\u{2B00}-\u{2BFF}\u{FE0F}\u{200D}\u{20E3}]+`);
  const EMOTICON = R(
    String.raw`(?<![A-Za-z0-9])(?:[:;=]['^-]?[DPpOo3|/\\*\]\[}{><]+|[)(\]\[][-^']?[:;=]|<3+|</3|\^_?\^|-_-|[oO]_[oO]|>_<|T_T|\\o/|:'[()])(?![A-Za-z0-9])` +
    String.raw`|[:;]['^-]?[)(]+` +
    String.raw`|(?<![A-Za-z0-9)])=['^-]?[)(]+|=['^-]?[)(]+(?![A-Za-z0-9(.$-])` +
    String.raw`|(?<=[A-Za-z]{2})<3+(?![.,]?\d)`);

  const PHONE_WORDS = new Set("call dial ring phone telephone tel mobile cell fax hotline text sms whatsapp".split(" "));
  const PHONE_LINKS = new Set("to on at or us me him her them back".split(" "));
  const PIN_WORDS = new Set("code pin extension ext passcode password zip postcode cvv cvc otp".split(" "));
  const STRONG_WORDS = new Set("number id reference ref serial account tracking booking confirmation".split(" "));
  const WEAK_WORDS = new Set(("order ticket invoice case claim policy receipt transaction parcel shipment container payment " +
    "reservation membership customer batch lot item job issue bug request").split(" "));
  const LINK_WORDS = new Set("is was are were ends ending in with starts starting begins beginning".split(" "));
  const NOT_COUNTED = new Set(("is was has does goes ships arrives departs leaves shows needs blocks remains includes contains " +
    "says reads starts ends closes opens runs takes comes this his its us as yes plus minus less").split(" "));
  const NUMBERED = new Set(("ref reference serial account order invoice tracking part model phone tel fax id room flight case " +
    "ticket policy licence license booking customer vat tax reg registration").split(" "));
  const DATE_BEFORE = new Set(("on dated due until till by since from before after for shipped delivered arriving arrives arrive " +
    "arrived expires expired ends ending starts starting begins beginning monday tuesday wednesday " +
    "thursday friday saturday sunday mon tue tues wed thu thur thurs fri sat sun").split(" "));
  const CLASSIC_FRACTIONS = new Set([2, 3, 4, 5, 6, 8, 10, 16, 32, 64, 100].map(BigInt));
  const RATIO_WORDS = new Set("ratio ratios scale odds aspect mix split proportion".split(" "));
  const DURATION_WORDS = new Set("lasted took duration runtime recording video clip track lap pace length long".split(" "));
  const CLOCK_WORDS = new Set(("window slot open opens opening hours shift session served service available booked appointment " +
    "delivery arrive arrives arriving").split(" "));
  const CAPITAL_WORDS = new Set(("IN ON AT TO OF BY OR IS AS IF NO SO WE HE MY ME DO GO UP AN AM PM BE IT US THE AND FOR BUT NOT " +
    "ALL").split(" "));
  const SHORT_WORDS = new Set("my to in on of go up no do me we us it is at by be so or an as if he hi ok".split(" "));
  const ZONES = "ET|CT|MT|PT|EST|EDT|CST|CDT|MST|MDT|PST|PDT|GMT|UTC|BST|CET|CEST|IST|AEST|AEDT|JST";
  const SIZES = "XXXL|XXL|XXS|XL|XS|S|M|L";
  const LENGTHS = "mm|cm|km|m|ft|in|inch|inches|feet|foot|metres?|meters?|yards?|yd|px|pixels?";
  const VERSIONED = ("version|ver|release|firmware|patch|kernel|build|python|node|nodejs|ios|ipados|macos|watchos|tvos|android|" +
    "chrome|firefox|safari|java|php|ruby|rust|swift|kotlin|golang|django|rails|react|angular|vue|docker|" +
    "kubernetes|ubuntu|debian|fedora|windows|linux|postgres|postgresql|mysql|redis|numpy|pytorch|tensorflow|cuda|" +
    "xcode|gcc|clang|unity|minecraft|typescript|bluetooth|usb|hdmi|opengl|vulkan|directx|" +
    "upgrade[ds]?|update[ds]?|downgrade[ds]?");
  const SEPARATORS = { ".": "dot", "/": "slash", "@": "at", "-": "dash", "_": "underscore", ":": "colon", "=": "equals", "&": "and",
    "+": "plus" };
  const HALF_DAY = String.raw`(?:\s?([AaPp])\.[Mm]\.?|\s?([AaPp])[Mm]\b)`;
  const AMOUNT = String.raw`(?P<whole>\d{1,3}(?:,\d{3})+|\d+)(?:\.(?P<fraction>\d+))?`;
  const MONEY = R(String.raw`(?:(?<![\w.])(?P<minus>-)\s?)?(?P<symbol>[$\u{a3}\u{20ac}\u{a5}\u{20b9}]|(?<![A-Za-z])(?:` + CURRENCY_CODES + String.raw`)(?![A-Za-z]))\s?` +
    String.raw`(?P<inner>-)?` + AMOUNT + String.raw`(?:\s?(?P<scale>thousand|million|billion|trillion|bn|[kKmMbB])(?![A-Za-z]))?`);
  const SIGN_AFTER = String.raw`[$\u{a3}\u{20ac}\u{a5}\u{20b9}](?![\w$\u{a3}\u{20ac}\u{a5}\u{20b9}])|(?:` + CURRENCY_CODES + String.raw`)(?![A-Za-z])`;
  const MONEY_AFTER = R(String.raw`(?:(?<![\w.])(?P<minus>-)\s?)?(?<![\w.,^])` +
    String.raw`(?P<whole>\d{1,3}(?:,\d{3})+|[1-9]\d{0,2}(?: \d{3})+|\d+)(?:\.(?P<fraction>\d+))?` +
    String.raw`\s?(?P<symbol>` + SIGN_AFTER + ")");
  const RANGE_BEFORE_SIGN = R(String.raw`(?<![\w.,-])(\d[\d,]*(?:\.\d+)?)-(?=\d[\d,]*(?:\.\d+)?\s?(?:` + SIGN_AFTER + "))");
  const CONTINENTAL = R(String.raw`(?<![\w.,])([1-9]\d{0,2}(?:\.\d{3})+),(\d{1,2})(?!\d|[.,]\d)`);
  const COMMA_AFTER_SIGN = R(String.raw`((?:[$\u{a3}\u{20ac}\u{a5}\u{20b9}]|(?<![A-Za-z])(?:` + CURRENCY_CODES + String.raw`))\s?-?\d+),(\d{1,2})(?!\d|[.,]\d)`);
  const COMMA_BEFORE_SIGN = R(String.raw`(?<![\w.,])(\d+),(\d{1,2})(?=\s?(?:%|` + SIGN_AFTER + "))");
  const FORMULA_END = R(String.raw`(?<![\w$])(?<!\d\s)\$(?:(?=[^\s\d$.,;:!?)])[^$\n]{0,80}` +
    String.raw`|(?=\d)[^$\n]{0,80}[=^_\\{}+*/<>][^$\n]{0,80}` +
    String.raw`|(?=\s)[^$\n]{0,80}[=^_\\{}+*<>][^$\n]{0,80})\d\s?\$(?![\w$])`);
  const VULGAR = R(String.raw`(?<![\d.,/])(?:(\d+)\s?)?([\u{bd}\u{2153}\u{2154}\u{bc}\u{be}\u{2155}\u{2156}\u{2157}\u{2158}\u{2159}\u{215a}\u{2150}\u{215b}\u{215c}\u{215d}\u{215e}\u{2151}\u{2152}])`);
  const ADDRESS = R(
    String.raw`(?P<url>(?:https?://|www\.)[^\s<>\"']+)` +
    String.raw`|(?P<mail>[\w.+-]+@[\w-]+(?:\.[\w-]+)+)` +
    String.raw`|(?P<file>(?<![\w@./-])(?P<folder>/?(?:[\w~-]+/)+)?(?P<stem>[\w-]+(?:\.[\w-]+)*)\.(?P<ending>[A-Za-z][A-Za-z0-9]{0,9})` +
    String.raw`(?![\w-])(?P<path>/[^\s<>\"']*)?)`);
  const DAY_RANGE = R(String.raw`\b(` + DAY_PATTERN + "|" + MONTH_PATTERN + String.raw`)\b(\.?)\s?[-\u{2013}]\s?(` + DAY_PATTERN + "|" +
    MONTH_PATTERN + String.raw`)\b(\.?)`);
  const RATIO = R(String.raw`(?<![\d:.,])(\d{1,3}):(\d{1,3}(?:,\d{3})+|\d+)(?![\d:]|[.,]\d)`);
  const CODE = R(String.raw`(?<![\w.,/-])(#)?([A-Za-z0-9]+(?:-[A-Za-z0-9]+)+)(?![\w-]|[.,]\d)`);
  const SLASHED = R(String.raw`(?<![\w.,/-])(\d+(?:/\d+){2,})(?![\w/]|[.,]\d)`);
  // Python's (?(lead)...) conditional, as two alternatives tried in the same order.
  const GROUPED_CORE = String.raw`[A-Z0-9]*\d[A-Z0-9]*(?: [A-Z0-9]*\d[A-Z0-9]*)+`;
  const GROUPED = R(String.raw`(?<![\w.,:/#+-])(?P<plus>\+)?(?:(?P<lead>[A-Z]{2}) (?P<core1>` + GROUPED_CORE + String.raw`)(?: (?P<trail>[A-Z]{2})(?![\w]))?` +
    String.raw`|(?P<core2>` + GROUPED_CORE + String.raw`))(?![\w]|[.,:/-]\d|\s?[%\u{b0}])`);
  const GLUED = R(String.raw`(?<![\w'.])(?=[A-Za-z0-9]*[A-Za-z])(?=[A-Za-z0-9]*\d)[A-Za-z0-9]+(?![\w']|\.\d)`);

  const ACCENT_DECIDES = { "résumé": "R EH1 Z AH0 M EY2", "resumé": "R EH1 Z AH0 M EY2", "exposé": "EH2 K S P OW0 Z EY1",
    "rosé": "R OW0 Z EY1", "pâté": "P AA0 T EY1", "maté": "M AA1 T EY2", "saké": "S AA1 K EY2", "lamé": "L AA0 M EY1" };
  const ACCENTED = R(String.raw`\b(` + Object.keys(ACCENT_DECIDES).join("|") + String.raw`)(s?)\b`, true);
  const LETTER_FORMS = { "ß": "ss", "æ": "ae", "Æ": "Ae", "œ": "oe", "Œ": "Oe", "ø": "o", "Ø": "O", "đ": "d", "ł": "l", "Ł": "L" };
  const MARK = /\p{Mn}/u, ASCII_ALPHA = /^[A-Za-z]+$/;

  function plainLetter(ch) {
    if (isascii(ch) || !isalpha(ch)) return ch;
    if (has(LETTER_FORMS, ch)) return LETTER_FORMS[ch];
    const stripped = [...ch.normalize("NFKD")].filter((c) => !MARK.test(c)).join("");
    return ASCII_ALPHA.test(stripped) ? stripped : ch;
  }

  // ---- numbers as words (BigInt: Python's integers have no limit) ----
  function cardinal(number) {
    number = BigInt(number);
    if (number < 0n) return "minus " + cardinal(-number);
    if (number < 20n) return ONES[Number(number)];
    if (number < 100n) {
      const tens = number / 10n, ones = number % 10n;
      return TENS[Number(tens) - 2] + (ones ? " " + ONES[Number(ones)] : "");
    }
    if (number < 1000n) {
      const hundreds = number / 100n, rest = number % 100n;
      return ONES[Number(hundreds)] + " hundred" + (rest ? " " + cardinal(rest) : "");
    }
    if (number >= 1000n ** BigInt(SCALES.length + 1)) return digits(number.toString());
    const parts = [];
    for (let power = SCALES.length; power > 0; power--) {
      const unit = 1000n ** BigInt(power);
      const count = number / unit;
      number %= unit;
      if (count) parts.push(cardinal(count) + " " + SCALES[power - 1]);
    }
    if (number) parts.push(cardinal(number));
    return parts.join(" ");
  }
  function ordinal(number) {
    const ws = cardinal(number).split(" ");
    const last = ws[ws.length - 1];
    ws[ws.length - 1] = ORDINAL[last] || (last.endsWith("y") ? last.slice(0, -1) + "ieth" : last + "th");
    return ws.join(" ");
  }
  function digits(text, zero = "zero") {
    const out = [];
    for (const c of text) if (isdigit(c)) out.push(c === "0" ? zero : ONES[Number(int(c))]);
    return out.join(" ");
  }
  function year(number) {
    number = BigInt(number);
    if (!(1000n <= number && number <= 2999n)) return cardinal(number);
    const high = number / 100n, low = number % 100n;
    if (low === 0n) return high % 10n === 0n ? cardinal(number) : cardinal(high) + " hundred";
    if (high === 20n && low < 10n) return "two thousand " + ONES[Number(low)];
    return cardinal(high) + (low < 10n ? " oh " + ONES[Number(low)] : " " + cardinal(low));
  }
  function pairs(number) {
    number = BigInt(number);
    const high = number / 100n, low = number % 100n;
    if (!(100n <= number && number <= 9999n) || (low === 0n && (high < 10n || high % 10n === 0n))) return cardinal(number);
    if (low === 0n) return cardinal(high) + " hundred";
    return cardinal(high) + (low < 10n ? " oh " + ONES[Number(low)] : " " + cardinal(low));
  }
  function numberWords(text) {
    const [whole, , fraction] = partition(replaceAll(text, ",", ""), ".");
    if (fraction) return (whole ? cardinal(int(whole)) : "") + " point " + digits(fraction);
    if ((plen(whole) > 1 && whole.startsWith("0")) || plen(whole) > 18) return digits(whole);
    return cardinal(int(whole));
  }
  function duration(hours, minutes, seconds) {
    const parts = [];
    for (const [count, name] of [[hours, "hour"], [minutes, "minute"], [seconds, "second"]]) {
      if (count) parts.push(`${cardinal(count)} ${BigInt(count) === 1n ? name : name + "s"}`);
    }
    if (parts.length < 2) return parts.length ? parts[0] : "zero seconds";
    return parts.slice(0, -1).join(", ") + " and " + parts[parts.length - 1];
  }
  function pluralOf(ws) {
    let [head, , last] = rpartition(ws, " ");
    last = last.endsWith("y") ? last.slice(0, -1) + "ies" : last + "s";
    return (head ? head + " " : "") + last;
  }
  function fractionWords(top, bottom, article = false) {
    top = BigInt(top); bottom = BigInt(bottom);
    const [one, many] = has(FRACTION, bottom.toString()) ? FRACTION[bottom.toString()] : [ordinal(bottom), ordinal(bottom) + "s"];
    if (top !== 1n) return `${cardinal(top)} ${many}`;
    return (article ? ("aeiou".includes(one[0]) ? "an " : "a ") : "one ") + one;
  }
  const versionPart = (part) => (plen(part) > 2 ? digits(part) : numberWords(part));
  const minor = (part) => (part.startsWith("0") ? digits(part, "oh") : cardinal(int(part)));
  function shortNumber(text) {
    if (text.startsWith("0")) return false;
    const chars = [...text];
    return chars.length <= 2 || (chars.length > 1 && chars.slice(1).every((c) => c === "0"));
  }
  const spelled = (letters) => `${SPELL}${letters}${END}`;
  const BEFORE_SPLIT = R(String.raw`[.!?;\u{2026}]\s|[()\"]`);
  const LETTERS = R("[A-Za-z]+");
  function before(text, position, count = 4) {
    const parts = split(BEFORE_SPLIT, text.slice(cpBack(text, position, 80), position));
    const clause = parts[parts.length - 1];
    const ws = findall(LETTERS, clause).map(lower);
    return ws.slice(Math.max(0, ws.length - count));
  }
  function identifier(text, position) {
    const ws = before(text, position);
    if (!ws.length) return null;
    const last = ws[ws.length - 1];
    if (PHONE_WORDS.has(last) || (PHONE_LINKS.has(last) && ws.slice(Math.max(0, ws.length - 3), ws.length - 1).some((w) => PHONE_WORDS.has(w)))) return "phone";
    let index = ws.length - 1;
    while (index > 0 && LINK_WORDS.has(ws[index])) index--;
    for (const [kind, group] of [["phone", PHONE_WORDS], ["pin", PIN_WORDS], ["strong", STRONG_WORDS], ["weak", WEAK_WORDS]]) {
      if (group.has(ws[index])) return kind;
    }
    return last === "enter" ? "enter" : null;
  }
  const DECIMAL_COMMA = RF(String.raw`\d+,\d{1,2}`);
  const decimalComma = (text) => (match(DECIMAL_COMMA, text) ? replaceAll(text, ",", ".") : text);
  function addressDigits(text) {
    const n = plen(text);
    if (n <= 2 && !(n === 2 && text[0] === "0")) return cardinal(int(text));
    if (n === 4 && 1900n <= int(text) && int(text) <= 2099n) return year(int(text));
    return digits(text);
  }

  // regexes of the rewriting passes
  const RE = {
    following: RM(String.raw`\s*([A-Za-z]+)`),
    followingOrDigit: RM(String.raw`\s*([A-Za-z]+|\d)`),
    firstWord: RM("[A-Za-z']+"),
    enDashRange: R(String.raw`(?<=\d)\u{2013}(?=\d)`),
    inline: R(String.raw`([A-Za-z']+)?\{([^{}]*)\}`),
    dots: R(String.raw`\.{3,}`),
    spacedMinus: R(String.raw`(?<=\d)\s-\s(?=\d[\d.,]*\s?=)`),
    joinedMinus: R(String.raw`(?<![\d.,/-])(\d[\d.,]*)-(?=\d[\d.,]*=)`),
    dashes: R(String.raw`-{2,}|\s-\s`),
    without: R(String.raw`\bw/o\b`, true),
    with: R(String.raw`\bw/(?=\s|$)`, true),
    notApplicable: R(String.raw`\bn/a\b`, true),
    compares: rx(String.raw`[<>\u{2265}\u{2264}\u{2260}]|!=`),
    digit: rx(String.raw`\d`),
    cents: R(String.raw`(?<![\w.])(\d[\d,]*(?:\.\d+)?)\s?\u{a2}`),
    sevenThirty: R(String.raw`(?<![\d.,:])(\d{1,2})\.([0-5]\d)(?=\s?[AaPp]\.?[Mm]\b)`),
    dotTimes: R(String.raw`(?<![\d.,:])(\d{1,2})\.(00|15|30|45)(\s?(?:to|until|till|and|[-\u{2014}])\s?)(\d{1,2})\.(00|15|30|45)(?!\d|\.\d)`),
    time: R(String.raw`(?<![\d:])(\d{1,2}):([0-5]\d)(?::([0-5]\d))?` + HALF_DAY + String.raw`?(?![\d:])`),
    hourHalfDay: R(String.raw`(?<![\d.:,])(\d{1,2})` + HALF_DAY + String.raw`(?![A-Za-z])`),
    seenDates: R(String.raw`(?<![\d/.])(\d{1,2})/(\d{1,2})(?:/(?:\d{4}|\d{2}))?(?![\d/])`),
    fullDate: rx(String.raw`(?<![\d/])\d{1,2}/\d{1,2}/(?:\d{4}|\d{2})(?![\d/])`),
    isoDate: R(String.raw`(?<![\d-])(\d{4})-(\d{1,2})-(\d{1,2})(?![\d-])`),
    slashIsoDate: R(String.raw`(?<![\d/])(\d{4})/(\d{1,2})/(\d{1,2})(?![\d/])`),
    dotDate: R(String.raw`(?<![\d.])(\d{1,2})\.(\d{1,2})\.(\d{4})(?!\d|\.\d)`),
    slashDate: R(String.raw`(?<![\d/])(\d{1,2})/(\d{1,2})/(\d{4}|\d{2})(?![\d/])`),
    monthDays: R(String.raw`\b(` + MONTH_PATTERN + String.raw`)\.?\s+(\d{1,2})(?:st|nd|rd|th)?\s?[-\u{2014}]\s?(\d{1,2})(?:st|nd|rd|th)?(?:,?\s+(\d{4}))?(?![\d:])`),
    daysMonth: R(String.raw`(?<![\d.,/:-])(\d{1,2})(?:st|nd|rd|th)?\s?[-\u{2014}]\s?(\d{1,2})(?:st|nd|rd|th)?\s+(` + MONTH_PATTERN + String.raw`)\b(?:,?\s+(\d{4}))?`),
    monthDay: R(String.raw`\b(` + MONTH_PATTERN + String.raw`)\.?\s+(\d{1,2})(?:st|nd|rd|th)?(?:,?\s+(\d{4}))?(?![\d:])`),
    dayMonth: R(String.raw`(?<![\d.,/:-])(\d{1,2})(?:st|nd|rd|th)?\s+(` + MONTH_PATTERN + String.raw`)\b(\.(?=\s+[a-z\d]))?(?:,?\s+(\d{4}))?`),
    phone: R(String.raw`(?<![\d-])(?:\+?(1)[-.\s])?\(?(\d{3})\)?[-.\s](\d{3})[-.\s](\d{4})(?![\d-])`),
    shortPhone: R(String.raw`(?<![\w.,-])(\d{3})-(\d{4})(?![\w-])`),
    version: R(String.raw`(?<![\w.])([vV])?(\d+(?:\.\d+){2,})(?:-(\d+)(?![\w-]|\.\d))?(?![\w])`),
    vVersion: R(String.raw`(?<![\w.])([vV])(\d+)\.(\d{2,3})(?!\d|\.\d|\w)`),
    named: R(String.raw`\b(` + VERSIONED + String.raw`)(\.?\s(?:(?:from|to)\s)?)(\d+)\.(\d{2,3})(?:(\s(?:to|or|and)\s)(\d+)\.(\d{2,3}))?(?!\d|\.\d|\s?%|\s(?:percent|million|billion)\b)`, true),
    minus: R(String.raw`(?<![\w.])-(?=\d)`),
    by: R(String.raw`(?:(?<=\d)\s?[xX\u{d7}]|\u{d7})\s?(?=\d[\d.,]*\s?(?:` + LENGTHS + String.raw`)\b)`),
    times: R(String.raw`(?<=\d)\s[xX\u{d7}]\s(?=\d)`),
    chat: R(String.raw`(?<![\w/'])[A-Za-z0-9/]*[0-9/][A-Za-z0-9/]*(?![\w/'])`),
    decades: R(String.raw`(?<![\w.])'?(\d{2}|\d{4})'?s\b`),
    ordinals: R(String.raw`(?<![\w.])(\d+)(?:st|nd|rd|th)\b`),
    resolution: R(String.raw`(?<![\w.])(144|240|360|480|540|576|720|900|1080|1440|2160|4320)([pi])\b`),
    percent: R(String.raw`(\d[\d,]*(?:\.\d+)?)\s?%`),
    mixed: R(String.raw`(?<![\d/.,-])(\d+) (\d)/(2|3|4|5|6|8|10|16)(?![\d/]|\.\d)`),
    fraction: R(String.raw`(?<![\d/.])(\d+)/(\d+)(?![\d/]|\.\d)`),
    years: R(String.raw`(?<![\w.,-])(\d{4})[-\u{2013}](\d{4})(?![\w.,-])`),
    range: R(String.raw`(?<![\w.,-])(\d+)[-\u{2013}](\d+)(?![\w-]|[.,]\d)`),
    timesX: R(String.raw`(?<![\w.])(\d+)x\b`),
    power: R(String.raw`(?<=\d)\^(?=\d)`),
    sizes: R(String.raw`(?<![\w-])(` + SIZES + ")-(" + SIZES + String.raw`)(?![\w-])`),
    numberSign: R(String.raw`#(?=\d)`),
    orderSign: R(String.raw`#(?=[A-Z]{1,4}-?\d)`),
    hashtag: R(String.raw`#(?=[A-Za-z])`),
    about: R(String.raw`~(?=\s?\d)`),
    comparisons: R(String.raw`(?<=[\w\s` + END + IS + String.raw`])(>=|<=|!=)(?=[\w\s` + SPELL + "])"),
    ti: R(String.raw`(?<=\d) Ti\b`),
    refNo: R(String.raw`\b([Rr])ef\.\s?[Nn]o\.(?=\s)`),
    dotted: R(String.raw`\b((?:[A-Za-z]\.){2,})`),
    abbreviation: R(String.raw`\b([A-Za-z]{1,6})\b(\.?)(?![A-Za-z'\u{2019}])`),
    numbers: R(String.raw`(?<![\w.])\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?|(?<![\w.])\.\d+`),
    slashWords: R(String.raw`(?<=[A-Za-z])/(?=[A-Za-z])`),
    lessGreater: R(String.raw`(?<=\d)` + IS + String.raw`?\s?[<>]\s?(?=\d)|(?<=\s)[<>](?=\s)`),
    symbols: R("[" + Object.keys(SYMBOL_WORDS).map(classChar).join("") + "]"),
    capitalBefore: rx(String.raw`[A-Z][a-z]+\s+$`),
    previousWord: rx(String.raw`([A-Za-z]+)\.?\s+$`),
    model: rx(String.raw`(?<![A-Za-z])([A-Z]{2,5}) $`),
    lettersAll: R("[A-Za-z]+"),
    counted: rx("(?:[^s]s|people|men|women|children|feet|miles)$"),
    runs: R("[A-Za-z]+|\\d+"),
    pieces: R("[A-Za-z]+|\\d+|."),
    addressParts: R("[A-Z]?[a-z]+|[A-Z]+(?![a-z])"),
    scheme: R(String.raw`^(?:https?://)?(?:www\.)?`),
    digitsLetters: RF(String.raw`\d+[A-Za-z]+`),
    ordinalGroup: RF(String.raw`\d+(?:s|st|nd|rd|th)`),
    wordEnd: rx("[A-Za-z']+$"),
    caps: R("[A-Za-z]{2,}"),
    triple: rx(String.raw`(.)\1\1`),
    stretched: R(String.raw`(.)\1{2,}`),
    liked: rx(String.raw`([lfsz])\1\1+$`),
    stretchedEnd: R(String.raw`(.)\1{2,}$`),
    capitals: RF(String.raw`[a-z]+(?:[A-Z][a-z]+)*[A-Z]{2,}(?:'s)?`),
    parts: R(String.raw`[A-Z]?[a-z']+|[A-Z]+(?![a-z])`),
    letterAhead: RM(String.raw`\s*[A-Za-z]`),
    hissAfter: RM(String.raw`[.!?\u{2026},;:\u{2014}\"')]*(?:\s|$)`),
    replyAfter: RM(String.raw`(?:[.?\u{2026},;\u{2014}]*[\"']?(?:\s|$)|!+[\"']?\s*$)`),
    tokens: R(SPELL + String.raw`([A-Za-z ]+)` + END + "|" + INLINE + "(.)" + INLINE +
      String.raw`|('?[A-Za-z]+(?:'[A-Za-z]+)*)|(\d+)|(-)|(\S)`),
  };
  const CLOCK = String.raw`\d{1,2}:[0-5]\d(?:\s?[AaPp]\.?[Mm]\.?)?|\b\d{1,2}\s?[AaPp]\.?[Mm]\.?`;
  RE.zone = R("(" + CLOCK + String.raw`)\s(` + ZONES + String.raw`)\b`);
  RE.clockRange = R("(" + CLOCK + String.raw`)\s?[-\u{2014}]\s?(?=\d{1,2}(?::[0-5]\d|\s?[AaPp]\.?[Mm]\b))`);

  function pyEscape(s) {                            // re.escape, for the order of the unit names
    let out = "";
    for (const ch of s) out += "()[]{}?*+-|^$\\.&~# \t\n\r\v\f".includes(ch) ? "\\" + ch : ch;
    return out;
  }

  class Normalizer {
    constructor(tables, isWord) {
      this.isWord = isWord;
      this.abbreviations = {};
      for (const [, columns] of tables.rows("abbreviations.txt")) this.abbreviations[columns[0]] = [columns[1], columns[2]];
      this.units = {};
      this.exactUnits = {};
      for (const [, columns] of tables.rows("units.txt")) {
        if (columns.length > 3 && columns[3] === "exact") this.exactUnits[columns[0]] = [columns[1], columns[2]];
        else this.units[lower(columns[0])] = [columns[1], columns[2]];
      }
      this.chat = {};
      this.replies = new Set();
      for (const [number, columns] of tables.rows("chat.txt")) {
        if (columns.length > 2 && !(columns.length === 3 && columns[2] === "alone")) throw new ValueError(`chat.txt line ${number}`);
        this.chat[columns[0]] = columns.length > 1 ? columns[1] : "-";
        if (columns.length > 2) this.replies.add(columns[0]);
      }
      this.endings = tables.mapping("endings.txt");
      const names = [...new Set(Object.keys(this.units).concat(Object.keys(this.exactUnits)).map(pyEscape))]
        .sort((a, b) => plen(b) - plen(a) || cmp(a, b));
      this.unitNames = names.join("|");
      this.unit = RD(String.raw`(?<![\w.])(\d[\d,]*(?:\.\d+)?)(?:[-\u{2013}](\d[\d,]*(?:\.\d+)?))?\s?(` + this.unitNames +
        String.raw`)(?![A-Za-z])(\.(?=\s+(?-i:[a-z])))?`, true);
      this.area = R(String.raw`(?<![\w.])(\d[\d,]*(?:\.\d+)?)\s?(km|cm|mm|m|ft|mi|yd|in)([\u{b2}\u{b3}])`);
      this.areaPlain = R(String.raw`(?<![\w.])(\d[\d,]*(?:\.\d+)?)\s?(km|cm|mm|m|ft)([23])(?![\w])`);
      this.unitAhead = RM(String.raw`\s?(?:` + this.unitNames + String.raw`)(?![A-Za-z])`, true);
      this.clockUnitAhead = RM(String.raw`\s?(?:%|percent|million|billion|` + this.unitNames + String.raw`)(?![A-Za-z])`, true);
      this.inline = [];
      this.dayFirst = false;
      this.dated = false;
    }

    amount(symbol, whole, fraction, scale) {
      const [one, many, cent, cents] = get(CURRENCY, symbol);
      if (scale) {
        scale = has(SCALE_LETTER, lower(scale)) ? SCALE_LETTER[lower(scale)] : lower(scale);
        return `${numberWords(whole + (fraction ? "." + fraction : ""))} ${scale} ${many}`;
      }
      if (fraction && (plen(fraction) > 2 || !cent)) return `${numberWords(whole + "." + fraction)} ${many}`;
      const amount = int(whole), change = int((fraction || "0").padEnd(2, "0"));
      const parts = [];
      if (amount || !change) parts.push(`${cardinal(amount)} ${amount === 1n ? one : many}`);
      if (change && cent) parts.push(`${cardinal(change)} ${change === 1n ? cent : cents}`);
      return parts.join(" and ");
    }
    static continental(m) {
      const gs = m.group(1).split(".");
      if (gs.length === 4 && gs.every((g) => int(g) <= 255n)) return m.group(0);
      return gs.join(",") + "." + m.group(2);
    }
    money(m) {
      const whole = replaceAll(replaceAll(m.group("whole"), ",", ""), " ", "");
      const groups = m.m.groups;
      const ws = this.amount(m.group("symbol"), whole, m.group("fraction"), groups.scale ?? null);
      return (groups.minus || groups.inner ? " minus " : " ") + ws + " ";
    }
    static halfDay(m, dotted, plain) {
      if (!(dotted || plain)) return "";
      const following = lstrip(m.string.slice(m.end()));
      const first = match(RE.firstWord, following);
      const closes = Boolean(dotted) && m.group(0).endsWith(".") && (
        !following || (isupper(firstCp(following)) && first !== null && SENTENCE_STARTERS.has(lower(first.group(0)))));
      return `${SPELL}${upper(dotted || plain)}M${END}${closes ? " ." : ""} `;
    }
    time(m) {
      const hour = int(m.group(1)), minute = int(m.group(2)), second = m.group(3);
      const half = m.group(4) || m.group(5);
      const prior = before(m.string, m.start(), 3);
      if (second !== null && !half && !(prior.length && prior[prior.length - 1] === "at")) return ` ${duration(hour, minute, int(second))} `;
      if (hour > 24n && second === null && !half) {
        const after = match(RE.following, m.string.slice(m.end()));
        if (prior.some((w) => DURATION_WORDS.has(w)) || (after && DURATION_WORDS.has(lower(after.group(1))))) return ` ${duration(0n, hour, minute)} `;
        return ` ${cardinal(hour)} ${minute < 10n ? "oh " : ""}${cardinal(minute)} `;
      }
      if (hour > 24n) return m.group(0);
      let ws = cardinal(hour);
      if (minute === 0n) {
        if (half) { /* nothing */ }
        else if (hour === 0n) ws = "midnight";
        else if (hour > 12n) ws += " hundred";
        else ws += " o'clock";
      } else {
        ws += (minute < 10n ? " oh " : " ") + cardinal(minute);
      }
      if (second) ws += " and " + cardinal(int(second)) + (int(second) === 1n ? " second" : " seconds");
      return ` ${ws} ` + Normalizer.halfDay(m, m.group(4), m.group(5));
    }
    dotTimes(m) {
      const following = m.string.slice(m.end());
      if (int(m.group(1)) > 23n || int(m.group(4)) > 23n || !before(m.string, m.start()).some((w) => CLOCK_WORDS.has(w))
          || match(this.clockUnitAhead, following)) return m.group(0);
      return `${m.group(1)}:${m.group(2)}${m.group(3)}${m.group(4)}:${m.group(5)}`;
    }
    ratio(m) {
      const first = m.group(1), second = m.group(2);
      if (plen(first) <= 2 && plen(second) === 2 && "012345".includes(second[0])) {
        const after = match(RE.following, m.string.slice(m.end()));
        if (!(before(m.string, m.start(), 3).some((w) => RATIO_WORDS.has(w)) || (after && RATIO_WORDS.has(lower(after.group(1)))))) {
          return m.group(0);
        }
      }
      return ` ${numberWords(first)} to ${numberWords(second)} `;
    }
    date(month, day, yearText = null, the = false) {
      month = BigInt(month); day = BigInt(day);
      if (!(1n <= month && month <= 12n) || !(1n <= day && day <= 31n)) return null;
      const name = MONTHS[Number(month) - 1];
      let ws = the ? `the ${ordinal(day)} of ${name}` : `${name} ${ordinal(day)}`;
      if (yearText) ws += " " + (plen(yearText) === 4 ? year(int(yearText)) : (int(yearText) < 10n ? "oh " : "") + cardinal(int(yearText)));
      return ` ${ws} `;
    }
    slashDate(m) {
      let first = int(m.group(1)), second = int(m.group(2));
      if ((first > 12n && 12n >= second) || (this.dayFirst && second <= 12n)) [first, second] = [second, first];
      return this.date(first, second, m.group(3)) || m.group(0);
    }
    dotDate(m) {
      let day = int(m.group(1)), month = int(m.group(2));
      if (month > 12n && 12n >= day) [day, month] = [month, day];
      return this.date(month, day, m.group(3)) || m.group(0);
    }
    dayMonth(m) {
      const month = m.group(2);
      const spoken = this.date(get(MONTH_NUMBER, lower(month).slice(0, 3)), int(m.group(1)), m.group(4), true);
      if (spoken === null) return m.group(0);
      return spoken + (m.group(3) && MONTHS.includes(month) ? ". " : "");
    }
    daysOfMonth(m) {
      const monthFirst = isalpha(firstCp(m.group(1)));
      let [month, first, second] = monthFirst ? [m.group(1), m.group(2), m.group(3)] : [m.group(3), m.group(1), m.group(2)];
      first = int(first); second = int(second);
      if (!(1n <= first && first <= 31n && 1n <= second && second <= 31n)) return m.group(0);
      const name = MONTHS[get(MONTH_NUMBER, lower(month).slice(0, 3)) - 1];
      const ws = monthFirst ? `${name} ${ordinal(first)} to ${ordinal(second)}` : `the ${ordinal(first)} to the ${ordinal(second)} of ${name}`;
      return ` ${ws}${m.group(4) ? " " + year(int(m.group(4))) : ""} `;
    }
    dayRange(m) {
      const names = [];
      for (const name of [m.group(1), m.group(3)]) {
        const key = lower(name);
        names.push(has(this.abbreviations, key) ? this.abbreviations[key][1] : has(DAY_NAMES, key) ? DAY_NAMES[key] : name);
      }
      const following = lstrip(m.string.slice(m.end()));
      const first = match(RE.firstWord, following);
      const ends = Boolean(m.group(4)) && (!following || (isupper(firstCp(following)) && first !== null && SENTENCE_STARTERS.has(lower(first.group(0)))));
      return `${names[0]} to ${names[1]}${ends ? " ." : ""}`;
    }
    shortDate(m) {
      const firstText = m.group(1), secondText = m.group(2);
      if (plen(firstText) > 2 || plen(secondText) > 2) return null;
      let first = int(firstText), second = int(secondText);
      const prior = before(m.string, m.start());
      if (1n <= first && first <= 12n && 20n <= second && second <= 49n && prior.includes("card")) {
        return ` ${MONTHS[Number(first) - 1]} ${year(2000n + second)} `;
      }
      const last = prior.length ? prior[prior.length - 1] : "";
      const classic = first < second && CLASSIC_FRACTIONS.has(second);
      const zero = (plen(firstText) === 2 && firstText[0] === "0") || (plen(secondText) === 2 && secondText[0] === "0");
      if (!(zero || (12n < first && first <= 31n && second <= 12n && !(first === 24n && second === 7n))
            || (DATE_BEFORE.has(last) && (!classic || last === "on"))
            || (this.dated && !classic && first !== second))) return null;
      if (first > 12n || (this.dayFirst && second <= 12n)) [first, second] = [second, first];
      return this.date(first, second);
    }
    fraction(m) {
      const spoken = this.shortDate(m);
      if (spoken) return spoken;
      const top = int(m.group(1)), bottom = int(m.group(2));
      if ((top === 24n && bottom === 7n) || top === bottom) return ` ${cardinal(top)} ${cardinal(bottom)} `;
      if (!(0n < top && top < 100n) || !(1n < bottom && bottom <= 100n)) return ` ${cardinal(top)} over ${cardinal(bottom)} `;
      return ` ${fractionWords(top, bottom)} `;
    }
    mixedFraction(m) {
      const top = int(m.group(2)), bottom = int(m.group(3));
      if (!(0n < top && top < bottom)) return m.group(0);
      return ` ${numberWords(m.group(1))} and ${fractionWords(top, bottom, true)} `;
    }
    vulgar(m) {
      const [top, bottom] = m.group(2).normalize("NFKC").split("⁄").map(int);
      if (m.group(1) === null) return ` ${top}/${bottom} `;
      return ` ${cardinal(int(m.group(1)))} and ${fractionWords(top, bottom, true)} `;
    }
    static phone(m) {
      const gs = m.groups().filter((g) => g);
      return " " + gs.map((g) => (g === "800" ? "eight hundred" : digits(g, "oh"))).join(", ") + " ";
    }
    measure(m) {
      const unit = m.group(3);
      const names = (has(this.exactUnits, unit) && this.exactUnits[unit]) || (has(this.units, lower(unit)) ? this.units[lower(unit)] : null);
      if (names === null || (plen(unit) === 1 && ["-", "'", "/"].includes(m.string.slice(m.end(3), m.end(3) + 1)))) return m.group(0);
      const [one, many] = names;
      const value = decimalComma(m.group(1)), up = m.group(2) && decimalComma(m.group(2));
      if (up) return ` ${numberWords(value)} to ${numberWords(up)} ${many} `;
      return ` ${numberWords(value)} ${value === "1" || value === "1.0" ? one : many} `;
    }
    square(m) {
      const unit = m.group(2);
      const [one, many] = unit === "in" ? ["inch", "inches"] : (has(this.exactUnits, unit) && this.exactUnits[unit]) || get(this.units, unit);
      const kind = "²2".includes(m.group(3)) ? "square" : "cubic";
      const value = decimalComma(m.group(1));
      return ` ${numberWords(value)} ${kind} ${value === "1" || value === "1.0" ? one : many} `;
    }
    counted(text, end) {
      const after = match(RE.following, text.slice(end));
      if (!after) return false;
      const word = lower(after.group(1));
      if (["people", "men", "women", "children", "feet", "miles"].includes(word)) return true;
      return word.length > 3 && word.endsWith("s") && !word.endsWith("ss") && !NOT_COUNTED.has(word) && this.isWord(word.slice(0, -1));
    }
    number(m, text) {
      const value = m.group(0);
      const plain = replaceAll(value, ",", "");
      if (!plain.includes(".") && !value.includes(",") && !plain.startsWith("0") && plen(plain) > 2) {
        const kind = identifier(text, m.start());
        const limit = { phone: 3, pin: 3, strong: 4, enter: 4, weak: 5 }[kind] ?? 7;
        if (plen(plain) >= limit && !(kind === null && plain.endsWith("000"))
            && !(kind === "enter" && plen(plain) === 4 && 1900n <= int(plain) && int(plain) <= 2099n)
            && !([null, "phone", "weak"].includes(kind) && this.counted(text, m.end()))) {
          return ` ${digits(plain, kind === "phone" ? "oh" : "zero")} `;
        }
      }
      if (!plain.includes(".") && !value.includes(",") && plen(plain) === 4 && !plain.startsWith("0")) {
        const n = int(plain);
        const prior = findall(RE.lettersAll, cpLast(text.slice(0, m.start()), 24));
        const after = match(RE.following, text.slice(m.end()));
        const counted = Boolean(after) && search(RE.counted, lower(after.group(1))) !== null;
        const dated = prior.length > 0 && (YEAR_CONTEXT.has(lower(prior[prior.length - 1])) || has(MONTH_NUMBER, lower(prior[prior.length - 1]).slice(0, 3)));
        const model = search(RE.model, text.slice(0, m.start()));
        if (model && !CAPITAL_WORDS.has(model.group(1)) && !counted) return ` ${pairs(n)} `;
        if ((1100n <= n && n <= 1999n) || (2000n <= n && n <= 2099n && (dated || !counted))) return ` ${year(n)} `;
      }
      return ` ${numberWords(value)} `;
    }

    readsAsWord(letters) {
      const low = lower(letters);
      if (low.length < 3 || !this.isWord(low)) return false;
      return low.length > 3 || ["s", "es", "ed", "ing"].some((ending) => this.isWord(low + ending));
    }
    static alnum(group, zero = "zero") {
      return findall(RE.runs, group).map((run) => (isdigit(run) ? digits(run, zero) : spelled(run))).join(" ");
    }
    static digitGroups(gs, zero = "zero") {
      if (gs.every((g) => shortNumber(g) && plen(g) <= 2)) return gs.map((g) => cardinal(int(g))).join(" , ");
      return gs.map((g) => digits(g, zero)).join(" , ");
    }
    codeWords(gs, m) {
      const token = gs.join("-");
      if (![...token].some((c) => isdigit(c))) return null;
      const kinds = gs.map((g) => (isdigit(g) ? "d" : [...g].some((c) => isdigit(c)) ? "m" : isupper(g) ? "u" : "l"));
      if (kinds.every((k) => k === "d")) {
        if (gs.length === 2) {
          const years = gs.every((g) => plen(g) === 4 && 1000n <= int(g) && int(g) <= 2100n);
          if (!(m.group(1) || gs.some((g) => plen(g) > 1 && g[0] === "0")
                || (Math.min(...gs.map(plen)) >= 4 && !years)
                || ["strong", "pin"].includes(identifier(m.string, m.start())))) return null;
        }
        return Normalizer.digitGroups(gs);
      }
      if (gs.length === 2 && kinds[1] === "d" && "ul".includes(kinds[0])) {
        const [letters, number] = gs;
        if ((kinds[0] === "l" && letters.length > 1) || this.readsAsWord(letters)) return null;
        if (shortNumber(number)) return `${spelled(letters)} ${cardinal(int(number))}`;
        return `${spelled(letters)} , ${digits(number)}`;
      }
      if (gs.length === 2 && kinds[0] === "d" && (kinds[1] === "l" || plen(gs[1]) === 1 || match(RE.digitsLetters, gs[1]))) return null;
      if (kinds.some((kind, k) => (kind === "l" && plen(gs[k]) > 2) || (kind === "m" && match(RE.ordinalGroup, gs[k])))) return null;
      if (![...token].some((c) => isupper(c)) && (kinds.includes("l") || kinds.includes("m"))) return null;
      const parts = [];
      gs.forEach((g, k) => {
        const kind = kinds[k];
        if (kind === "d") parts.push(digits(g));
        else if (kind === "m") parts.push(Normalizer.alnum(g));
        else if (plen(g) > 3 && this.isWord(lower(g))) parts.push(g);
        else parts.push(plen(g) <= 5 ? spelled(g) : g);
      });
      return parts.join(" , ");
    }
    code(m) {
      const ws = this.codeWords(m.group(2).split("-"), m);
      if (ws === null) return m.group(0);
      return (m.group(1) ? " number " : " ") + ws + " ";
    }
    grouped(m) {
      const text = m.string, plus = m.group("plus");
      let lead = m.group("lead"), trail = m.group("trail");
      const core = (m.group("core1") ?? m.group("core2")).split(" ");
      let head = "", tail = "";
      if (lead !== null && CAPITAL_WORDS.has(lead)) {
        head = lead + " "; tail = trail ? " " + trail : ""; lead = null; trail = null;
      }
      const kind = identifier(text, m.start());
      const pure = core.every((part) => isdigit(part));
      if (pure && !plus && !lead) {
        const following = text.slice(m.end());
        const amount = this.counted(text, m.end()) || match(this.unitAhead, following) !== null;
        if (kind === null && core[0][0] !== "0" && plen(core[0]) <= 3 && core.slice(1).every((part) => plen(part) === 3)
            && (core[core.length - 1] === "000" || (amount && core.length > 2))) return head + core.join(",") + tail;
        if (!(kind || (plen(core[0]) > 1 && core[0][0] === "0") || (core.length > 2 && core.some((part) => plen(part) > 2)))) return m.group(0);
      } else if (!pure && !lead && !plus && kind === null) {
        return m.group(0);
      }
      const zero = plus || kind === "phone" || (pure && core[0][0] === "0" && core.reduce((s, part) => s + plen(part), 0) >= 10) ? "oh" : "zero";
      let spoken = pure ? Normalizer.digitGroups(core, zero)
        : core.map((part) => (isdigit(part) ? digits(part, zero) : Normalizer.alnum(part, zero))).join(" , ");
      if (lead) spoken = `${spelled(lead)} , ${spoken}`;
      if (trail) spoken = `${spoken} , ${spelled(trail)}`;
      return head + (plus ? " plus " : " ") + spoken + " " + tail;
    }
    glued(m) {
      const out = [];
      for (const run of findall(RE.runs, m.group(0))) {
        if (isdigit(run)) out.push(shortNumber(run) ? cardinal(int(run)) : digits(run));
        else if (run.length === 1 || ((isupper(run) || run.length <= 3) && !this.readsAsWord(run))) out.push(spelled(run));
        else out.push(run);
      }
      return " " + out.join(" ") + " ";
    }

    addressPart(part) {
      const low = lower(part);
      if (part.length > 1 && this.isWord(low) && (part.length > 2 || SHORT_WORDS.has(low))) return part;
      return part.length <= 3 ? spelled(part) : part;
    }
    spokenAddress(core) {
      const pieces = findall(RE.pieces, core);
      const mail = core.includes("@");
      let host = false;
      const out = [];
      pieces.forEach((piece, index) => {
        const following = index + 1 < pieces.length ? pieces[index + 1] : "";
        if (piece === "@") host = true;
        if (isdigit(piece)) out.push(addressDigits(piece));
        else if (!isalpha(piece)) out.push(has(SEPARATORS, piece) ? SEPARATORS[piece] : "");
        else if (index && pieces[index - 1] === "." && (host || !mail) && (
          ["", "/", ":", "?", "#"].includes(following)
          || (plen(piece) <= 3 && (has(this.endings, lower(piece)) || !this.isWord(lower(piece)))))) {
          out.push(has(this.endings, lower(piece)) && this.endings[lower(piece)] === "word" ? lower(piece) : spelled(piece));
        } else {
          let letters = "";
          for (const part of findall(RE.addressParts, piece)) {
            const spoken = this.addressPart(part);
            if (spoken.startsWith(SPELL)) { letters += part; continue; }
            if (letters) { out.push(spelled(letters)); letters = ""; }
            out.push(spoken);
          }
          if (letters) out.push(spelled(letters));
        }
      });
      return out.filter((item) => item).join(" ");
    }
    address(m) {
      const whole = m.group(0);
      const text = rstrip(whole, ".,;:!?)");
      const tail = whole.slice(text.length);
      if (m.group("file")) {
        const ending = m.group("ending");
        const known = has(this.endings, lower(ending));
        const marked = Boolean(m.group("folder") || m.group("path")) || m.group("stem").includes("_");
        if (!(known || (marked && plen(ending) <= 5 && (islower(ending) || isupper(ending)) && !this.isWord(lower(ending))))) return whole;
      }
      const core = rstrip(sub(RE.scheme, "", text), "/");
      return ` ${this.spokenAddress(core)} ${tail}`;
    }
    chatForm(m) {
      const key = lower(m.group(0));
      if (!has(this.chat, key)) return m.group(0);
      const said = this.chat[key];
      if (said === "-") return " ";
      return " " + words(said).map((part) => (part.startsWith("*") ? `${SPELL}${part.slice(1)}${END}` : part)).join(" ") + " ";
    }
    inlinePronunciation(m) {
      const symbols = words(m.group(2));
      if (!symbols.length || symbols.every((symbol) => plen(symbol) === 1 && isascii(symbol))) return m.group(0);
      let phones;
      try {
        phones = isArpabet(symbols) ? convert("", symbols) : fromText(symbols.join(" "));
      } catch (error) {
        if (error instanceof ValueError) return m.group(0);
        throw error;
      }
      this.inline.push([m.group(1) || "", phones]);
      return ` ${INLINE}${String.fromCodePoint(INLINE_BASE + this.inline.length - 1)}${INLINE} `;
    }
    dotted(m, text) {
      const letters = replaceAll(m.group(0), ".", "");
      const key = lower(letters);
      const following = lstrip(text.slice(m.end()));
      const ws = key === "eg" ? "for example" : key === "ie" ? "that is" : `${SPELL}${upper(letters)}${END}`;
      const first = match(RE.firstWord, following);
      const endsSentence = !following || (isupper(firstCp(following)) && first && SENTENCE_STARTERS.has(lower(first.group(0))));
      return ` ${ws}${endsSentence && m.group(0).endsWith(".") ? " ." : ""} `;
    }
    abbreviation(m, text) {
      const token = m.group(1), stop = m.group(2);
      const key = lower(token);
      const following = text.slice(m.end());
      const nextWord = match(RE.followingOrDigit, following);
      const capitalNext = Boolean(nextWord) && isupper(firstCp(nextWord.group(1)));
      const capitalBefore = search(RE.capitalBefore, text.slice(0, m.start())) !== null;
      if ((key === "st" || key === "dr") && isupper(token[0])) {
        if (capitalNext && (stop || key === "dr")) return key === "st" ? " saint " : " doctor ";
        if (capitalBefore && stop) return (key === "st" ? " street" : " drive") + (!strip(following) ? " ." : " ");
        return m.group(0);
      }
      if (!has(this.abbreviations, key)) return m.group(0);
      let [when, ws] = this.abbreviations[key];
      const numberNext = Boolean(nextWord) && isdigit(nextWord.group(1));
      if (when === "title") {
        if (!(isupper(token[0]) && (capitalNext || stop))) return m.group(0);
      } else if (when === "number") {
        const previous = search(RE.previousWord, text.slice(0, m.start()));
        if (!(stop && (numberNext || (key === "no" && previous && NUMBERED.has(lower(previous.group(1))))))) return m.group(0);
      } else if (!stop || (when === "small" && isupper(token))) {
        return m.group(0);
      } else if (["mar", "sun", "sat", "jan", "dec", "max", "min", "est", "no"].includes(key) && !isupper(token[0]) && !numberNext) {
        if (!(["max", "min", "est"].includes(key) && nextWord && islower(firstCp(nextWord.group(1))))) return m.group(0);
      }
      if (key === "est" && !numberNext) ws = "estimated";
      const endsSentence = when !== "title" && (!strip(following) || (capitalNext && SENTENCE_STARTERS.has(lower(nextWord.group(1)))));
      return ` ${ws}${endsSentence ? " ." : ""} `;
    }

    rewrite(text) {
      text = sub(VULGAR, (m) => this.vulgar(m), text);
      text = sub(this.area, (m) => this.square(m), text);
      text = replaceAll(replaceAll(text, "²", " squared "), "³", " cubed ");
      text = replaceAll(replaceAll(text.normalize("NFKC"), "⁄", "/"), "−", " minus ");
      if (!isascii(text)) {
        text = sub(ACCENTED, (m) => "{" + get(ACCENT_DECIDES, lower(m.group(1))) + (m.group(2) ? " Z" : "") + "}", text);
        let plain = "";
        for (const ch of text) plain += plainLetter(ch);
        text = plain;
      }
      text = sub(RE.enDashRange, "-", text);
      text = sub(DAY_RANGE, (m) => this.dayRange(m), text);
      for (const [old, replacement] of [["‘", "'"], ["’", "'"], ["‚", "'"], ["‛", "'"], ["`", "'"], ["´", "'"], ["“", '"'], ["”", '"'],
        ["„", '"'], ["‟", '"'], ["«", '"'], ["»", '"'], ["–", "—"], ["―", "—"], ["[", "("], ["]", ")"]]) {
        text = replaceAll(text, old, replacement);
      }
      this.inline = [];
      text = sub(RE.inline, (m) => this.inlinePronunciation(m), text);
      text = replaceAll(replaceAll(text, "{", "("), "}", ")");
      text = sub(ADDRESS, (m) => this.address(m), text);
      text = replaceAll(text, "→", " to ");
      text = sub(EMOJI, " ", text);
      text = sub(EMOTICON, " ", text);
      text = sub(RE.dots, "…", text);
      text = sub(RE.spacedMinus, " minus ", text);
      text = sub(RE.joinedMinus, (m) => m.group(1) + " minus ", text);
      text = sub(RE.dashes, "—", text);
      text = sub(RE.without, " without ", text);
      text = sub(RE.with, " with ", text);
      text = sub(RE.notApplicable, " not applicable ", text);
      text = replaceAll(text, "'n'", " 'n ");
      if (search(RE.compares, text)) text = this.markComparisons(text);
      if (search(RE.digit, text)) text = this.figures(text);
      return this.signs(text);
    }

    figures(text) {
      text = sub(FORMULA_END, (m) => m.group(0).slice(0, -1) + FORMULA_SIGN, text);
      text = sub(CONTINENTAL, (m) => Normalizer.continental(m), text);
      text = sub(COMMA_AFTER_SIGN, (m) => m.group(1) + "." + m.group(2), text);
      text = sub(COMMA_BEFORE_SIGN, (m) => m.group(1) + "." + m.group(2), text);
      text = sub(MONEY, (m) => this.money(m), text);
      text = sub(RANGE_BEFORE_SIGN, (m) => m.group(1) + " to ", text);
      text = sub(MONEY_AFTER, (m) => this.money(m), text);
      text = sub(RE.cents, (m) => ` ${numberWords(m.group(1))} ${m.group(1) === "1" ? "cent" : "cents"} `, text);

      text = sub(RE.zone, (m) => `${m.group(1)} ${spelled(m.group(2))}`, text);
      text = sub(RE.sevenThirty, (m) => m.group(1) + ":" + m.group(2), text);
      text = sub(RE.dotTimes, (m) => this.dotTimes(m), text);
      text = sub(RE.clockRange, (m) => m.group(1) + " to ", text);
      text = sub(RATIO, (m) => this.ratio(m), text);
      text = sub(RE.time, (m) => this.time(m), text);
      text = sub(RE.hourHalfDay, (m) => ` ${cardinal(int(m.group(1)))} ` + Normalizer.halfDay(m, m.group(2), m.group(3)), text);

      let seen = findall(RE.seenDates, text).map(([a, b]) => [int(a), int(b)]);
      seen = seen.filter(([a, b]) => !(a === 24n && b === 7n));
      this.dayFirst = seen.some(([a, b]) => a > 12n && 12n >= b) && !seen.some(([a, b]) => b > 12n && 12n >= a);
      this.dated = search(RE.fullDate, text) !== null;
      text = sub(RE.isoDate, (m) => this.date(int(m.group(2)), int(m.group(3)), m.group(1)) || m.group(0), text);
      text = sub(RE.slashIsoDate, (m) => this.date(int(m.group(2)), int(m.group(3)), m.group(1)) || m.group(0), text);
      text = sub(RE.dotDate, (m) => this.dotDate(m), text);
      text = sub(RE.slashDate, (m) => this.slashDate(m), text);
      text = sub(RE.monthDays, (m) => this.daysOfMonth(m), text);
      text = sub(RE.daysMonth, (m) => this.daysOfMonth(m), text);
      text = sub(RE.monthDay, (m) => this.date(get(MONTH_NUMBER, lower(m.group(1)).slice(0, 3)), int(m.group(2)), m.group(3)) || m.group(0), text);
      text = sub(RE.dayMonth, (m) => this.dayMonth(m), text);

      text = sub(RE.phone, Normalizer.phone, text);
      text = sub(RE.shortPhone, Normalizer.phone, text);
      text = sub(RE.version, (m) => " " + (m.group(1) ? spelled(m.group(1)) + " " : "")
        + m.group(2).split(".").map(versionPart).join(" point ")
        + (m.group(3) ? " dash " + numberWords(m.group(3)) : "") + " ", text);
      text = sub(RE.vVersion, (m) => ` ${spelled(m.group(1))} ${cardinal(int(m.group(2)))} point ${minor(m.group(3))} `, text);
      text = sub(RE.named, (m) => `${m.group(1)}${m.group(2)}${cardinal(int(m.group(3)))} point ${minor(m.group(4))}`
        + (m.group(5) ? `${m.group(5)}${cardinal(int(m.group(6)))} point ${minor(m.group(7))}` : "") + " ", text);
      text = sub(CODE, (m) => this.code(m), text);
      text = sub(SLASHED, (m) => " " + Normalizer.digitGroups(m.group(1).split("/")) + " ", text);
      text = sub(GROUPED, (m) => this.grouped(m), text);

      text = sub(RE.minus, " minus ", text);
      text = sub(RE.by, " by ", text);
      text = sub(RE.times, " times ", text);
      text = sub(RE.chat, (m) => this.chatForm(m), text);
      text = sub(RE.decades, (m) => " " + pluralOf(plen(m.group(1)) === 4 ? year(int(m.group(1))) : cardinal(int(m.group(1)))) + " ", text);
      text = sub(RE.ordinals, (m) => ` ${ordinal(int(m.group(1)))} `, text);
      text = sub(RE.resolution, (m) => ` ${pairs(int(m.group(1)))} ${spelled(m.group(2))} `, text);
      text = sub(RE.percent, (m) => ` ${numberWords(m.group(1))} percent `, text);
      text = sub(this.areaPlain, (m) => this.square(m), text);
      text = sub(this.unit, (m) => this.measure(m), text);
      text = sub(RE.mixed, (m) => this.mixedFraction(m), text);
      text = sub(RE.fraction, (m) => this.fraction(m), text);
      text = sub(RE.years, (m) => ` ${year(int(m.group(1)))} to ${year(int(m.group(2)))} `, text);
      text = sub(RE.range, (m) => ` ${numberWords(m.group(1))} to ${numberWords(m.group(2))} `, text);
      text = sub(RE.timesX, (m) => ` ${numberWords(m.group(1))} times `, text);
      text = sub(RE.power, " to the power of ", text);
      return text;
    }

    static namedLetters(text) {
      if (![..."=<>+*/^()≤≥≠≈×÷±"].some((sign) => text.includes(sign))) return text;
      const plain = withoutAngles(text);
      const places = new Set();
      for (const pattern of LETTER_PLACES) for (const m of finditer(pattern, plain)) places.add(m.start("letter"));
      if (!places.size) return text;
      let out = "";
      for (let index = 0; index < text.length; index++) out += places.has(index) ? `${SPELL}${upper(text[index])}${END}` : text[index];
      return out;
    }
    static markComparisons(text) {
      const plain = withoutAngles(text);
      const mark = (m) => {
        const prior = plain.slice(0, m.start()), after = plain.slice(m.end());
        const value = search(VALUE_BEFORE, prior);
        if (plain[m.start("sign")] !== m.group("sign")[0] || value === null || !match(VALUE_AFTER, after)) return m.group(0);
        let start = value.start();
        if (value.group(0) === ")" || value.group(0) === "]") {
          let depth = 0;
          while (start >= 0) {
            depth += (")]".includes(prior[start]) ? 1 : 0) - ("([".includes(prior[start]) ? 1 : 0);
            if (depth === 0) break;
            start -= 1;
          }
          while (start > 0) {
            const ch = cpBefore(prior, start);
            if (!(isalnum(ch) || ch === "_" || ch === ".")) break;
            start -= ch.length;
          }
        }
        const word = search(RE.wordEnd, rstrip(prior.slice(0, Math.max(start, 0))));
        if (word && !CLAUSE_OPENERS.has(lower(word.group(0)))) return m.group(0);
        return IS + m.group(0);
      };
      return sub(COMPARED, mark, text);
    }
    markComparisons(text) { return Normalizer.markComparisons(text); }
    static compared(m, ws) {
      const marked = m.group(0).startsWith(IS) || rstrip(m.string.slice(0, m.start()), " ").endsWith(IS);
      return ` ${marked ? "is " : ""}${ws} `;
    }
    joinedComparisons(text) {
      const plain = withoutAngles(text);
      const read = (m) => {
        const prior = rstrip(text.slice(0, m.start()), IS), after = text.slice(m.end());
        if (plain[m.start()] !== m.group(0)) return m.group(0);
        if (!(search(FIGURE_BEFORE, prior) || match(FIGURE_AFTER, after) || (search(LETTER_BEFORE, prior) && match(LETTER_AFTER, after)))) return m.group(0);
        return Normalizer.compared(m, m.group(0) === "<" ? "less than" : "greater than");
      };
      return sub(JOINED_COMPARISON, read, text);
    }
    signs(text) {
      text = Normalizer.namedLetters(text);
      if (text.includes("<") || text.includes(">")) text = this.joinedComparisons(text);
      text = sub(RE.sizes, (m) => ` ${spelled(m.group(1))} to ${spelled(m.group(2))} `, text);
      text = sub(RE.numberSign, " number ", text);
      text = sub(RE.orderSign, " number ", text);
      text = sub(RE.hashtag, " hashtag ", text);
      text = sub(RE.about, " about ", text);
      text = sub(RE.comparisons, (m) => Normalizer.compared(m, COMPARISONS[m.group(1)]), text);
      text = sub(RE.ti, ` ${spelled("TI")}`, text);
      text = sub(RE.refNo, (m) => m.group(1) + "eference number", text);
      if (text.includes(".")) { const t = text; text = sub(RE.dotted, (m) => this.dotted(m, t), text); }
      { const t = text; text = sub(RE.abbreviation, (m) => this.abbreviation(m, t), text); }
      if (search(RE.digit, text)) {
        text = sub(GLUED, (m) => this.glued(m), text);
        const t = text;
        text = sub(RE.numbers, (m) => this.number(m, t), text);
      }
      text = sub(RE.slashWords, " ", text);
      text = sub(RE.lessGreater, (m) => Normalizer.compared(m, m.group(0).includes("<") ? "less than" : "greater than"), text);
      text = sub(RE.symbols, (m) => ("≥≤≠".includes(m.group(0)) ? Normalizer.compared(m, SYMBOL_WORDS[m.group(0)]) : ` ${SYMBOL_WORDS[m.group(0)]} `), text);
      return replaceAll(text, IS, "");
    }

    // -- words --
    static unstretched(word) {
      if (!search(RE.triple, word)) return [];
      const cut = (doubleInside, doubleFinal) => sub(RE.stretched, (m) => {
        const final = m.end() === word.length;
        return m.group(1).repeat((final ? doubleFinal : doubleInside) ? 2 : 1);
      }, word);
      const liked = Boolean(search(RE.liked, word)) || (word.length > 0 && [...word].every((c) => c === "m"))
        || sub(RE.stretchedEnd, (m) => m.group(1), word) === "to";
      const ordered = [cut(true, liked), cut(true, !liked), cut(false, liked), cut(false, !liked)];
      return [...new Set(ordered)];
    }
    static chatWords(chat, raw) {
      if (chat === "-") return [];
      return words(chat).map((part) => (part.startsWith("*") ? word(lower(part.slice(1)), { raw, spell: true }) : word(lower(part), { raw: part })));
    }
    static reply(out, place) {
      const marks = [];
      for (const step of [-1, 1]) {
        let index = place + step;
        while (index >= 0 && index < out.length && out[index].punct && "\"'()".includes(out[index].mark)) index += step;
        if (!(index >= 0 && index < out.length)) marks.push("");
        else if (!out[index].punct) return false;
        else {
          marks.push(out[index].mark);
          const beyond = index + step >= 0 && index + step < out.length ? out[index + step] : null;
          if (out[index].mark === "," && beyond && !beyond.punct && beyond.text.length === 1 && beyond.text !== "i") return false;
        }
      }
      return !(marks[0] === "," && marks[1] === ",");
    }
    wordTokens(raw, allCaps, previous, following = true, fresh = false) {
      const low = lower(raw);
      const chat = has(this.chat, low) ? this.chat[low] : null;
      if (chat !== null && (raw.length > 1 || islower(raw)) && !this.replies.has(low)) return Normalizer.chatWords(chat, raw);
      if (has(ROMAN, raw) && previous && isupper(previous[0])) {
        const number = ROMAN[raw];
        const ws = CARDINAL_AFTER.has(lower(previous)) ? cardinal(number) : "the " + ordinal(number);
        return ws.split(" ").map((w) => word(w, { raw }));
      }
      const letters = replaceAll(raw, "'", "");
      const possessive = low.endsWith("'s");
      const stem = possessive ? raw.slice(0, -2) : raw;
      if (stem.length === 1 && isalpha(stem)) {
        const label = stem === "A" && previous && (!following || (!fresh && LABELLED.has(lower(previous))));
        if (["a", "A", "I", "i"].includes(stem) && !possessive && !label) return [word(lower(stem), { raw })];
        return [word(low, { raw, spell: true })];
      }
      if (stem.length > 2 && "SsZz".includes(stem[0]) && new Set(low).size === 1 && (allCaps || !isupper(stem))) {
        return [word(low, { raw, phones: [[low[0], 0]] })];
      }
      if (isupper(stem) && isalpha(stem) && stem.length >= 2 && stem.length <= 5 && (!allCaps || !this.isWord(lower(stem)))) {
        if (INITIALISMS.has(stem) || !this.isWord(lower(stem))) return [word(low, { raw, spell: true })];
      }
      if (!isalpha(letters)) return [];
      const capitals = match(RE.capitals, raw) !== null;
      if (capitals || !this.isWord(low)) {
        for (const candidate of capitals ? [] : Normalizer.unstretched(low)) {
          if (this.isWord(candidate)) return [word(candidate, { raw })];
        }
        const parts = findall(RE.parts, raw);
        if (parts.length > 1 && parts.join("") === raw) {
          const out = [];
          for (const part of parts) {
            const named = capitals && isupper(part) ? INITIALISMS.has(part) : false;
            const spelt = part.length === 1 || named || (isupper(part) && !this.isWord(lower(part)));
            if (capitals && out.length && out[out.length - 1].spell && (spelt || part === "'s")) {
              const last = out[out.length - 1];
              out[out.length - 1] = { ...last, text: last.text + lower(part), raw: last.raw + part };
            } else {
              out.push(word(lower(part), { attach: out.length > 0, raw: part, spell: spelt }));
            }
          }
          return out;
        }
      }
      return [word(low, { raw })];
    }

    tokens(text) {
      text = this.rewrite(text);
      const runs = findall(RE.caps, text);
      const allCaps = runs.length >= 3 && runs.filter((run) => isupper(run)).length >= 0.6 * runs.length;
      const out = [];
      let attach = false, previous = "", fresh = true;
      const short = [], replies = [];
      for (const m of finditer(RE.tokens, text)) {
        const [spelt, inline, w0, number, hyphen, other] = m.groups();
        let w = w0;
        const prior = m.start() ? cpBefore(text, m.start()) : " ";
        if (hyphen) {
          attach = out.length > 0 && !out[out.length - 1].punct && !isspace(prior) && m.end() < text.length && isalnum(cpAt(text, m.end()));
          continue;
        }
        let fresh_tokens = [];
        if (spelt !== null) {
          fresh_tokens = [word(lower(replaceAll(spelt, " ", "")), { raw: spelt, spell: true })];
        } else if (inline !== null) {
          const at = inline.codePointAt(0) - INLINE_BASE;
          const entry = this.inline[at < 0 ? this.inline.length + at : at];       // Python list indexing
          if (!entry) throw new Error("IndexError: list index out of range");
          const [written, phones] = entry;
          fresh_tokens = [word(lower(written), { raw: written, phones })];
        } else if (w !== null) {
          if (w.startsWith("'") && !LEADING_APOSTROPHE.has(lower(w))) {
            out.push(punct("'"));
            w = w.slice(1);
          } else if (w.startsWith("'") && !this.isWord(lower(w))) {
            w = w.slice(1);
          }
          fresh_tokens = this.wordTokens(w, allCaps, previous, match(RE.letterAhead, text.slice(m.end())) !== null, fresh);
          previous = w;
          if (["Ss", "ss", "Zz", "zz"].includes(w) && fresh_tokens.length === 1 && (isspace(prior) || "\"'(".includes(prior))
              && match(RE.hissAfter, text.slice(m.end()))) short.push(out.length);
          if (this.replies.has(lower(w)) && (w.length > 1 || islower(w)) && fresh_tokens.length === 1
              && (isspace(prior) || "\"'".includes(prior)) && match(RE.replyAfter, text.slice(m.end()))) replies.push(out.length);
        } else if (number !== null) {
          fresh_tokens = numberWords(number).split(" ").filter((x) => x).map((part) => word(part, { raw: number }));
        } else if (PUNCT_SET.has(other)) {
          if (!(other === "'" && isalnum(prior) && m.end() < text.length && isalnum(cpAt(text, m.end())))) out.push(punct(other));
          if (".!?;…".includes(other)) fresh = true;
          attach = false;
          continue;
        }
        if (fresh_tokens.length && attach) fresh_tokens[0] = { ...fresh_tokens[0], attach: true };
        out.push(...fresh_tokens);
        attach = false;
        fresh = fresh && !fresh_tokens.length;
      }
      for (const place of short) {
        let first = place, last = place + 1;
        while (first && (!out[first - 1].punct || "\"'()".includes(out[first - 1].mark))) first--;
        while (last < out.length && (!out[last].punct || "\"'()".includes(out[last].mark))) last++;
        if (out.slice(first, last).filter((t) => !t.punct).every((t) => SOUNDS.has(t.text) || (t.phones !== null && new Set(t.text).size === 1))) {
          out[place] = { ...out[place], phones: [[out[place].text[0], 0]] };
        }
      }
      for (const place of replies.slice().reverse()) {
        if (Normalizer.reply(out, place)) {
          const raw = out[place].raw;
          out.splice(place, 1, ...Normalizer.chatWords(get(this.chat, lower(raw)), raw));
        }
      }
      return out;
    }
  }
  const word = (text, { attach = false, raw = "", spell = false, phones = null } = {}) => ({ punct: false, text, attach, raw, spell, phones });
  const punct = (mark) => ({ punct: true, mark });

  // ---- engine.py ---------------------------------------------------------------------------------------
  const LEIGH = R(String.raw`(?<=[a-z]{2})leigh(?=s?$)`);
  const VIRUS = ["viruses", "virus"];

  class Phonemizer {
    constructor(data) {
      const tables = new Tables(data.tables);
      this.lexicon = data.lexicon;                   // .has(word), .get(word) -> phones
      this.aligner = new Aligner(data.align);
      this.model = data.model;
      this.homographs = new Homographs(data.homographs, get(data.tables, "homograph_rules.txt"));
      this.functions = tables.functionWords();
      this.phrases = tables.phrases();
      this.phraseLength = Math.max(1, ...[...this.phrases.keys()].map((key) => key.split(" ").length));
      this.exceptions = tables.pronunciations("exceptions.txt");
      this.names = tables.names("exceptions.txt");
      this.nameStarts = new Set([...this.names.keys()].map((key) => key.split(" ")[0]));
      this.nameLength = Math.max(1, ...[...this.names.keys()].map((key) => key.split(" ").length));
      this.userNames = new Set();
      this.overrides = new Map();
      this.letterNames = null;
      this.tables = tables;
      this.normalizer = new Normalizer(tables, (w) => this.isWord(w));
      this.cache = new Map();
    }
    entry(w, [kind, value]) {
      if (kind === "phones") return value.map((p) => p.slice());
      return convert(w, value, this.aligner.align(w, value));
    }
    known(w) {
      if (this.overrides.has(w)) return this.entry(w, this.overrides.get(w));
      if (this.exceptions.has(w)) return this.entry(w, this.exceptions.get(w));
      return this.lexicon.get(w);
    }
    ordinary(w) {
      if (w.length < 3 || !["s", "es", "ed", "ing"].some((ending) => this.lexicon.has(w + ending))) return null;
      return this.known(w);
    }
    guess(w) {
      if (!this.model) return this.spell(w);
      const spelling = this.model.spellable(w);
      const symbols = this.model.predict(spelling);
      const stressed = (s) => "012".includes(s[s.length - 1]);
      symbols.forEach((symbol, k) => {
        if (symbol === "R" && ![k - 1, k + 1].some((j) => j >= 0 && j < symbols.length && stressed(symbols[j]))) symbols[k] = "ER0";
      });
      if (!symbols.some(stressed)) return this.spell(w);
      return promote(convert(spelling, symbols, this.aligner.align(spelling, symbols)));
    }
    spell(letters) {
      if (this.letterNames === null) {
        this.letterNames = new Map(Object.entries(this.tables.mapping("letters.txt")).map(([key, value]) => [key, fromText(value)]));
      }
      const possessive = lower(letters).endsWith("'s");
      const names = [...lower(possessive ? letters.slice(0, -2) : letters)].filter((c) => this.letterNames.has(c)).map((c) => this.letterNames.get(c));
      const out = [];
      names.forEach((name, index) => {
        for (const [phone, stress] of name) out.push([phone, stress && index < names.length - 1 ? 2 : stress]);
      });
      return possessive && out.length ? plural(out) : out;
    }
    isWord(w) { return this.lexicon.has(w) || this.functions.has(w) || this.exceptions.has(w) || this.overrides.has(w); }
    fromParts(w) {
      if (w.endsWith("'s") && w.length > 3) {
        const found = this.fromParts(w.slice(0, -2));
        return found && [plural(found[0]), found[1]];
      }
      if (search(LEIGH, w)) return this.word(sub(LEIGH, "ley", w));
      for (const tail of VIRUS) {
        const second = w.endsWith(tail) && w.length - tail.length > 2 ? this.known(tail) : null;
        if (second !== null && second !== undefined) {
          const [first, source] = this.word(w.slice(0, -tail.length));
          const phones = promote(settleStress(compound(first, second)));
          return [phones, source === "model" ? "model" : "ending"];
        }
      }
      return null;
    }
    word(w) {
      if (this.cache.has(w)) return this.cache.get(w);
      let phones = this.known(w), source = "dictionary";
      if (phones === null) {
        phones = derive(w, (s) => this.known(s), (s) => this.ordinary(s));
        source = "ending";
        if (phones !== null) phones = settleStress(phones);
      }
      if (phones === null) {
        const found = this.fromParts(w);
        if (found) [phones, source] = found;
      }
      if (phones === null && w.endsWith("'s") && w.length > 3) { phones = plural(this.guess(w.slice(0, -2))); source = "model"; }
      if (phones === null) { phones = this.guess(w); source = "model"; }
      const result = [phones, source];
      this.cache.set(w, result);
      return result;
    }

    units(ws, places, written) {
      const units = [];
      let index = 0;
      while (index < ws.length) {
        const token = ws[index];
        if (token.phones !== null) {
          units.push(new Unit(token.text, token.phones.map((p) => p.slice()), "p", null, token.attach, "inline"));
        } else if (token.spell) {
          units.push(new Unit(token.text, this.spell(token.text), "p", null, token.attach, "letters"));
        } else {
          let size = this.nameStarts.has(token.text) ? this.nameSize(ws, index) : 0;
          if (size) {
            const key = ws.slice(index, index + size).map((t) => t.text);
            const entries = this.names.get(key.join(" "));
            key.forEach((text, position) => units.push(new Unit(text, this.entry(text, entries[position]), "p", null, token.attach && position === 0, "name")));
            index += size;
            continue;
          }
          size = this.phraseSize(ws, index);
          if (size) {
            const key = ws.slice(index, index + size).map((t) => t.text);
            const [kind, forms, final] = this.phrases.get(key.join(" "));
            if (forms.length === 1) units.push(new Unit(key.join(" "), forms[0], kind, final, token.attach, "phrase"));
            else key.forEach((text, position) => units.push(new Unit(text, forms[position], kind, null, token.attach && position === 0, "phrase")));
            index += size;
            continue;
          }
          if (this.overrides.has(token.text)) {
            units.push(new Unit(token.text, this.known(token.text), "c", null, token.attach, "override"));
          } else if (this.homographs && written && this.homographs.covers(token.text) && !this.exceptions.has(token.text)) {
            const [text, ending] = this.homographs.pronounce(token.text, written, places[index], token.attach);
            const phones = ending ? plural(fromText(text)) : fromText(text);
            units.push(new Unit(token.text, phones, "c", null, token.attach, "homograph"));
          } else if (this.functions.has(token.text)) {
            const [kind, form, final] = this.functions.get(token.text);
            units.push(new Unit(token.text, form, kind, final, token.attach, "function"));
          } else {
            const [phones, source] = this.word(token.text);
            units.push(new Unit(token.text, phones, "c", null, token.attach, source));
          }
        }
        index += 1;
      }
      return units;
    }
    nameSize(ws, index) {
      for (let size = Math.min(this.nameLength, ws.length - index); size > 1; size--) {
        const span = ws.slice(index, index + size);
        const key = span.map((t) => t.text).join(" ");
        if (this.names.has(key) && span.every((t) => t.phones === null && !t.spell) && !span.slice(1).some((t) => t.attach)
            && (this.userNames.has(key) || !span.some((t) => this.overrides.has(t.text)))) return size;
      }
      return 0;
    }
    phraseSize(ws, index) {
      for (let size = Math.min(this.phraseLength, ws.length - index); size > 1; size--) {
        const span = ws.slice(index, index + size);
        if (span.slice(1).every((t) => t.phones === null && !t.spell && !t.attach) && span[0].phones === null
            && !span[0].spell && this.phrases.has(span.map((t) => t.text).join(" "))
            && !span.some((t) => this.overrides.has(t.text))) return size;
      }
      return 0;
    }
    analyse(text) {
      const stream = this.normalizer.tokens(text);
      const written = stream.map((t) => (t.punct ? t.mark : t.raw || t.text || "x"));
      const out = [], ws = [], places = [];
      const flush = () => {
        if (ws.length) {
          const units = this.units(ws, places, written);
          stressClause(units);
          joinClause(units);
          out.push(["clause", units]);
          ws.length = 0;
          places.length = 0;
        }
      };
      stream.forEach((token, place) => {
        if (token.punct) { flush(); out.push(["punct", token.mark]); }
        else { ws.push(token); places.push(place); }
      });
      flush();
      return out;
    }
    pieces(text) {
      return this.analyse(text).map(([kind, value]) => (kind === "punct" ? [true, value] : [false, groups(value)]));
    }
    phonemize(text) {
      if (typeof text !== "string" || !strip(text)) throw new ValueError("text must not be empty");
      return tokenize(this.pieces(text));
    }
  }

  // ---- the data bundle ---------------------------------------------------------------------------------
  // "DIZ1", header length (u32), header JSON, then 4-byte aligned binary sections the header points to.
  function parseBundle(bytes) {
    const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    const magic = String.fromCharCode(...bytes.subarray(0, 4));
    if (magic !== "DIZ1") throw new Error("not a dizione data bundle");
    const length = view.getUint32(4, true);
    const header = JSON.parse(new TextDecoder().decode(bytes.subarray(8, 8 + length)));
    const section = (name) => { const [offset, size] = header.sections[name]; return bytes.subarray(offset, offset + size); };
    return { header, section };
  }

  class Lexicon {
    // Words sorted and front-coded; pronunciations as phone codes, front-coded against the previous entry.
    constructor(header, section) {
      const symbols = header.lexicon.symbols.map((text) => (text ? fromText(text)[0] : null));
      const count = header.lexicon.count;
      const wordPrefix = section("lexicon.word_prefix"), wordSuffix = section("lexicon.word_suffix");
      const phonePrefix = section("lexicon.phone_prefix"), phoneSuffix = section("lexicon.phone_suffix");
      this.index = new Map();
      this.starts = new Uint32Array(count + 1);
      this.codes = new Uint8Array(header.lexicon.codes);
      this.symbols = symbols;
      const decoder = new TextDecoder();
      let w = "", ws = 0, ps = 0, written = 0, previous = this.codes.subarray(0, 0);
      for (let k = 0; k < count; k++) {
        let end = wordSuffix.indexOf(0, ws);
        w = w.slice(0, wordPrefix[k]) + decoder.decode(wordSuffix.subarray(ws, end));
        ws = end + 1;
        this.index.set(w, k);
        end = phoneSuffix.indexOf(0, ps);
        const keep = phonePrefix[k];
        this.starts[k] = written;
        this.codes.set(previous.subarray(0, keep), written);
        this.codes.set(phoneSuffix.subarray(ps, end), written + keep);
        const size = keep + end - ps;
        previous = this.codes.subarray(written, written + size);
        written += size;
        ps = end + 1;
      }
      this.starts[count] = written;
    }
    has(w) { return this.index.has(w); }
    get(w) {
      const k = this.index.get(w);
      if (k === undefined) return null;
      const out = [];
      for (let j = this.starts[k]; j < this.starts[k + 1]; j++) out.push(this.symbols[this.codes[j]].slice());
      return out;
    }
  }

  function readModel(header, section) {
    const weights = {};
    for (const [name, spec] of Object.entries(header.g2p.arrays)) {
      const raw = section("g2p." + name);
      const data = new Float32Array(spec.shape.reduce((a, b) => a * b, 1));
      if (spec.dtype === "f32") {
        new Uint8Array(data.buffer).set(raw);
      } else {
        throw new Error("unknown weight type " + spec.dtype);
      }
      weights[name] = { data, shape: spec.shape };
    }
    return new Model(header.g2p.settings, weights);
  }

  function build(bytes) {
    const { header, section } = parseBundle(bytes);
    const phonemizer = new Phonemizer({
      tables: header.tables,
      align: header.align,
      homographs: header.homographs,
      lexicon: new Lexicon(header, section),
      model: readModel(header, section),
    });
    return {
      version: header.version,
      // {ids, stress, text}; null when there is nothing to pronounce; throws on any other failure.
      phonemize(text) {
        try {
          return phonemizer.phonemize(text);
        } catch (error) {
          if (!(error instanceof ValueError)) throw error;
          if (phonemizer.pieces(text).some(([mark, ws]) => !mark && ws.some((w) => w.length))) throw error;
          return null;
        }
      },
      // [true, mark] for punctuation and [false, [[[phone, stress], ...], ...]] for a run of words, as dizione.pieces.
      pieces: (text) => phonemizer.pieces(text),
      analyse: (text) => phonemizer.analyse(text),
      tokens: (text) => phonemizer.normalizer.tokens(text),
      word: (w) => phonemizer.word(w),
      g2p: (w) => phonemizer.model.predict(w),
      phonemizer,
    };
  }

  async function load(options = {}) {
    if (typeof options === "string" || options instanceof URL) options = { base: options };
    let bytes = options.bundle;
    if (!bytes || typeof bytes === "string" || bytes instanceof URL) {
      const here = (typeof document !== "undefined" && document.currentScript && document.currentScript.src) ||
                   (typeof location !== "undefined" ? location.href : undefined);
      const url = bytes ? new URL(bytes, here) : new URL(DATA_FILE, options.base ? new URL(options.base, here) : here);
      const response = await (options.fetch || fetch)(url.href);
      if (!response.ok) throw new Error(`${url.href}: ${response.status}`);
      bytes = await response.arrayBuffer();
    }
    if (bytes instanceof ArrayBuffer) bytes = new Uint8Array(bytes);
    return build(bytes);
  }

  root.Dizione = { load, build, DATA_FILE, ValueError, version: "0.1.0", _internals: { rx, RE, sub, Normalizer } };
})(typeof self !== "undefined" ? self : globalThis);
