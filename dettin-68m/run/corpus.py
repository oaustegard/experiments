#!/usr/bin/env python3
# ruff: noqa: C408, BLE001, B023, EXE001
"""Corpus builder for dettin-68m: open-access life-science article sections.

    python3 run/corpus.py sample --n-articles N --seed S   -> data/articles.jsonl
    python3 run/corpus.py fetch  --workers 6                -> out/xml/<pmcid>.xml.gz
    python3 run/corpus.py parse                             -> out/sections.jsonl.gz,
                                                               data/corpus_stats.json

Stdlib + requests only.  Every step is resumable: `sample` skips dates already
recorded in data/sample_dates.jsonl (its date stream and per-date picks are
deterministic in --seed, so a larger --n-articles later extends the same
sample), `fetch` skips XML already on disk, `parse` is a pure function of the
XML on disk and is simply rerun.
"""
import argparse
import datetime as dt
import gzip
import hashlib
import html.entities
import json
import os
import random
import re
import sys
import threading
import time
import xml.etree.ElementTree as ET
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "out"
ARTICLES = DATA / "articles.jsonl"
SAMPLE_DATES = DATA / "sample_dates.jsonl"
STATS = DATA / "corpus_stats.json"
XML_DIR = OUT / "xml"
FAILURES = OUT / "fetch_failures.jsonl"
SECTIONS = OUT / "sections.jsonl.gz"

UA = "corpus-builder/1.0 (research corpus; python-requests)"
EPMC_SEARCH = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
EPMC_FULLTEXT = "https://www.ebi.ac.uk/europepmc/webservices/rest/{}/fullTextXML"
NCBI_EFETCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

DATE_MIN = dt.date(2019, 1, 1)
DATE_MAX = dt.date(2025, 6, 30)
PER_DATE_MAX = 12
PER_JOURNAL_PER_DATE = 3

# Source `fetch` uses unless --source says otherwise.  Measured 2026-11-10 from
# the CCotw container, 40 articles at 6 workers: Europe PMC fullTextXML (one
# article per call) 0.61 articles/s; NCBI efetch db=pmc, 20 ids per call at
# <=3 calls/s, 36-46 articles/s.  Failures on one source can be retried from the
# other: `fetch --source epmc` only touches articles with no XML on disk.
DEFAULT_SOURCE = "ncbi"

# ---------------------------------------------------------------------------
# HTTP plumbing: global rate limiter, thread-local sessions, jittered retries
# ---------------------------------------------------------------------------


class RateLimiter:
    """Spaces request *starts* at least 1/rps seconds apart, across threads."""

    def __init__(self, rps):
        self.interval = 1.0 / rps
        self.lock = threading.Lock()
        self.next = 0.0

    def wait(self):
        with self.lock:
            now = time.monotonic()
            t = max(now, self.next)
            self.next = t + self.interval
        if t > now:
            time.sleep(t - now)


class FetchError(Exception):
    pass


_local = threading.local()
_jitter = random.Random()  # OS-seeded; never touches the seeded sampling RNGs


def _session():
    s = getattr(_local, "s", None)
    if s is None:
        s = _local.s = requests.Session()
        s.headers["User-Agent"] = UA
    return s


def http_get(url, params=None, limiter=None, validate=None, tries=6, timeout=90):
    """GET with retries.  `validate(response)` returns the value to hand back or
    raises ValueError (retried like a 5xx).  404/400 are permanent."""
    last = "unknown"
    for k in range(tries):
        retry_after = None
        if limiter:
            limiter.wait()
        try:
            r = _session().get(url, params=params, timeout=timeout)
            if r.status_code == 200:
                try:
                    return validate(r) if validate else r
                except ValueError as e:
                    last = f"invalid response: {e}"
            elif r.status_code in (400, 404):
                raise FetchError(f"HTTP {r.status_code}")
            else:
                last = f"HTTP {r.status_code}"
                retry_after = r.headers.get("Retry-After")
        except (requests.ConnectionError, requests.Timeout,
                requests.exceptions.ChunkedEncodingError) as e:
            last = type(e).__name__
        if k < tries - 1:
            delay = min(60.0, 2.0 ** k) * (0.5 + _jitter.random())
            if retry_after and retry_after.isdigit():
                delay = max(delay, min(int(retry_after), 120))
            time.sleep(delay)
    raise FetchError(f"{last} after {tries} tries")


def say(msg):
    print(msg, flush=True)


# ---------------------------------------------------------------------------
# 1. sample
# ---------------------------------------------------------------------------


def read_jsonl(path):
    if not path.exists():
        return []
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass  # a half-written last line from a killed run
    return rows


def append_jsonl(path, rows):
    with open(path, "a", encoding="utf-8") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
        f.flush()
        os.fsync(f.fileno())


def date_stream(seed):
    """Uniform random dates in [DATE_MIN, DATE_MAX], without repeats."""
    rng = random.Random(seed)
    n_days = (DATE_MAX - DATE_MIN).days + 1
    seen = set()
    while len(seen) < n_days:
        d = (DATE_MIN + dt.timedelta(days=rng.randrange(n_days))).isoformat()
        if d not in seen:
            seen.add(d)
            yield d


def _validate_search(r):
    try:
        j = r.json()
        j["hitCount"], j["resultList"]["result"]
    except (ValueError, KeyError, TypeError) as e:
        raise ValueError(f"search payload: {e}")
    return j


def query_date(d, seed, limiter):
    """All hits for one first-publication date, then the seeded pick."""
    q = ('OPEN_ACCESS:y AND HAS_FT:y AND LICENSE:"cc by" AND '
         f'PUB_TYPE:"research-article" AND FIRST_PDATE:{d}')
    hits, cursor = {}, "*"
    while True:
        j = http_get(EPMC_SEARCH, limiter=limiter, validate=_validate_search,
                     params=dict(query=q, format="json", resultType="lite",
                                 pageSize=100, cursorMark=cursor))
        res = j["resultList"]["result"]
        for x in res:
            if x.get("pmcid"):
                hits[x["pmcid"]] = x
        nxt = j.get("nextCursorMark")
        if not res or not nxt or nxt == cursor or len(hits) >= int(j["hitCount"]):
            break
        cursor = nxt
    # Server order is not stable, so sort before the seeded shuffle.
    cand = [hits[k] for k in sorted(hits)]
    random.Random(f"{seed}:{d}").shuffle(cand)
    picked, per_journal = [], Counter()
    for x in cand:
        journal = x.get("journalTitle") or x.get("journalIssn") or ""
        if per_journal[journal] >= PER_JOURNAL_PER_DATE:
            continue
        per_journal[journal] += 1
        picked.append(dict(pmcid=x["pmcid"], journal=x.get("journalTitle", ""),
                           year=int(x["pubYear"]) if str(x.get("pubYear", "")).isdigit() else None,
                           title=(x.get("title") or "").strip(),
                           pub_type=x.get("pubType", ""), date=d))
        if len(picked) == PER_DATE_MAX:
            break
    return len(hits), picked


def cmd_sample(a):
    DATA.mkdir(parents=True, exist_ok=True)
    have = {r["pmcid"] for r in read_jsonl(ARTICLES)}
    done = {r["date"] for r in read_jsonl(SAMPLE_DATES)}
    limiter = RateLimiter(5.0)
    stream = date_stream(a.seed)
    failed_dates = 0
    say(f"sample: have {len(have)} articles, {len(done)} dates done, target {a.n_articles}")
    while len(have) < a.n_articles:
        chunk = []
        while len(chunk) < a.workers:
            d = next(stream, None)
            if d is None:
                break
            if d not in done:
                chunk.append(d)
        if not chunk:
            say("sample: date range exhausted")
            break
        with ThreadPoolExecutor(a.workers) as ex:
            futs = {d: ex.submit(query_date, d, a.seed, limiter) for d in chunk}
        for d in chunk:  # apply in stream order so the cut at N is deterministic
            try:
                n_hits, picked = futs[d].result()
            except FetchError as e:
                failed_dates += 1
                say(f"sample: {d} failed ({e}); will be retried next run")
                continue
            fresh = [r for r in picked if r["pmcid"] not in have]
            take = fresh[: a.n_articles - len(have)]
            if take:
                append_jsonl(ARTICLES, take)
                have.update(r["pmcid"] for r in take)
            if len(take) == len(fresh):  # not cut short by N: the date is complete
                append_jsonl(SAMPLE_DATES, [dict(date=d, hits=n_hits, picked=len(picked))])
                done.add(d)
            say(f"sample: {d} hits={n_hits} picked={len(picked)} new={len(take)} "
                f"total={len(have)}/{a.n_articles}")
            if len(have) >= a.n_articles:
                break
    say(f"done -> data/articles.jsonl ({len(have)} articles)")


# ---------------------------------------------------------------------------
# 2. fetch
# ---------------------------------------------------------------------------


def xml_path(pmcid):
    return XML_DIR / f"{pmcid}.xml.gz"


def write_xml(pmcid, raw):
    tmp = XML_DIR / f"{pmcid}.xml.gz.part"
    with gzip.open(tmp, "wb", compresslevel=6) as f:
        f.write(raw)
    os.replace(tmp, xml_path(pmcid))


def _validate_article_xml(r):
    raw = r.content
    if b"<article" not in raw[:5000] and b"<article" not in raw:
        raise ValueError("no <article> element")
    if not raw.rstrip().endswith(b"</article>"):
        raise ValueError("truncated: no closing </article>")
    return raw


def fetch_epmc(pmcid):
    raw = http_get(EPMC_FULLTEXT.format(pmcid), validate=_validate_article_xml)
    write_xml(pmcid, raw)


_ARTICLE_RE = re.compile(rb"<article[\s>].*?</article>", re.DOTALL)
_ID_RE = re.compile(rb'<article-id[^>]*pub-id-type="(?:pmcid|pmc|pmc-uid)"[^>]*>\s*(?:PMC)?(\d+)\s*<')
_NCBI_HEAD = b'<?xml version="1.0" encoding="UTF-8"?>\n'


def split_efetch(raw):
    """{numeric pmc id: bytes of one standalone <article>} from an efetch reply."""
    out = {}
    for m in _ARTICLE_RE.finditer(raw):
        chunk = m.group(0)
        idm = _ID_RE.search(chunk)
        if idm:
            out[idm.group(1).decode()] = _NCBI_HEAD + chunk
    return out


def _validate_efetch(r):
    if not r.content.rstrip().endswith(b"</pmc-articleset>"):
        raise ValueError("truncated efetch reply")
    return r.content


def fetch_ncbi_batch(pmcids, limiter, tries=6):
    """One efetch call for several ids.  Returns {pmcid: error or None}.  If the
    call fails outright, the ids are retried in halves so one poisonous id
    cannot take its whole batch down."""
    nums = [p[3:] for p in pmcids]
    try:
        raw = http_get(NCBI_EFETCH, limiter=limiter, tries=tries,
                       params=dict(db="pmc", id=",".join(nums), retmode="xml",
                                   tool="corpus-builder"),
                       validate=_validate_efetch)
    except FetchError as e:
        if len(pmcids) == 1:
            return {pmcids[0]: str(e)}
        mid = len(pmcids) // 2
        res = fetch_ncbi_batch(pmcids[:mid], limiter, tries=3)
        res.update(fetch_ncbi_batch(pmcids[mid:], limiter, tries=3))
        return res
    got = split_efetch(raw)
    res = {}
    for p in pmcids:
        if p[3:] in got:
            write_xml(p, got[p[3:]])
            res[p] = None
        else:
            res[p] = "not in efetch reply"
    return res


def cmd_fetch(a):
    arts = read_jsonl(ARTICLES)
    if not arts:
        sys.exit("fetch: data/articles.jsonl is missing or empty; run `sample` first")
    XML_DIR.mkdir(parents=True, exist_ok=True)
    pmcids = list(dict.fromkeys(r["pmcid"] for r in arts))
    if a.limit:
        pmcids = pmcids[: a.limit]
    todo = [p for p in pmcids if not xml_path(p).exists()]
    source = a.source or DEFAULT_SOURCE
    say(f"fetch: {len(pmcids)} articles, {len(pmcids) - len(todo)} on disk, "
        f"{len(todo)} to fetch from {source} with {a.workers} workers")
    ok = failed = 0
    t0 = time.monotonic()

    def log_failure(p, err):
        append_jsonl(FAILURES, [dict(pmcid=p, source=source, error=err,
                                     ts=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"))])

    def tick(n_done):
        if n_done % 100 == 0:
            rate = n_done / max(time.monotonic() - t0, 1e-9)
            say(f"fetch: {n_done}/{len(todo)} (ok {ok}, failed {failed}) {rate:.2f} articles/s")

    with ThreadPoolExecutor(a.workers) as ex:
        if source == "epmc":
            futs = {ex.submit(fetch_epmc, p): p for p in todo}
            for fut in as_completed(futs):
                p = futs[fut]
                try:
                    fut.result()
                    ok += 1
                except Exception as e:  # one article never stops the run
                    failed += 1
                    log_failure(p, f"{type(e).__name__}: {e}")
                tick(ok + failed)
        elif source == "ncbi":
            limiter = RateLimiter(3.0)
            batches = [todo[i:i + a.batch] for i in range(0, len(todo), a.batch)]
            futs = {ex.submit(fetch_ncbi_batch, b, limiter): b for b in batches}
            for fut in as_completed(futs):
                try:
                    res = fut.result()
                except Exception as e:
                    res = {p: f"{type(e).__name__}: {e}" for p in futs[fut]}
                for p, err in res.items():
                    before = ok + failed
                    if err is None:
                        ok += 1
                    else:
                        failed += 1
                        log_failure(p, err)
                    if (ok + failed) // 100 > before // 100:
                        tick((ok + failed) // 100 * 100)
        else:
            sys.exit(f"fetch: unknown source {source!r}")
    n_ok = sum(1 for p in pmcids if xml_path(p).exists())
    say(f"done -> out/xml ({n_ok} ok, {len(pmcids) - n_ok} failed)")


# ---------------------------------------------------------------------------
# 3. parse
# ---------------------------------------------------------------------------

MIN_WORDS = 120          # a unit shorter than this is dropped
SPLIT_ABOVE = 700        # a unit longer than this is split into parts
PART_MAX = 600           # words per part
MIN_PARA_WORDS = 4       # a shorter paragraph is dropped as an equation remnant
EXPAND_ABOVE = 700       # a top-level sec with child secs and more words than this is emitted per child

BACKMATTER = ("data availability", "ethics", "author contribution", "funding", "conflict",
              "competing interest", "acknowledg", "abbreviation", "supplementary", "publisher",
              "consent", "additional information")

# Elements removed with their content (their tail text is kept).
DROP = {"table-wrap", "table-wrap-group", "fig", "fig-group", "disp-formula", "disp-formula-group",
        "supplementary-material", "fn", "graphic", "inline-graphic", "media", "label", "target",
        "ref-list", "caption", "title", "sec-meta", "fn-group", "permissions", "table"}
MATH = {"math", "alternatives", "inline-formula"}
MATH_SKIP = {"tex-math", "annotation", "annotation-xml", "graphic", "inline-graphic"}
BLOCKISH = {"p", "list", "list-item", "def-list", "def-item", "def", "term", "disp-quote", "statement"}
CITE_XREFS = {"bibr", "fn", "aff", "corresp", "author-notes", "table-fn"}

SENT = ""  # marks where a citation <xref> was removed
_SEP = r"[\s,;–—\-]"
_BR = r"[\(\[]" + _SEP + "*" + SENT + _SEP + r"*[\)\]]"
_CONNECT = (r"(?:see|also|e\.g\.|i\.e\.|cf\.|reviewed in|reviewed by|refs?\.?|references?|"
            r"described in|reported in|detailed in|in|by)")
_RE_MERGE = re.compile(SENT + "(?:" + _SEP + "*" + SENT + ")+")
_RE_BRCHAIN = re.compile(_BR + r"(?:\s*[,;–—\-]\s*" + _BR + ")*")
_RE_BRCONN = re.compile(r"[\(\[]\s*(?:" + _CONNECT + r"[\s,]*)+" + SENT + _SEP + r"*[\)\]]", re.IGNORECASE)
_RE_LEAD = re.compile(r"(?<=[\(\[])[\s,;]*" + SENT + r"[\s,;]*")
_RE_TRAIL = re.compile(r"[\s,;]*" + SENT + r"[\s,;]*(?=[\)\]])")
_RE_MID = re.compile(r"[,;]\s*" + SENT + r"(?=\s*[,;])")
_RE_WS = re.compile(r"\s+")
_RE_INVIS = re.compile("[\u200b‌‍⁠﻿­]")


def ln(tag):
    return tag.rsplit("}", 1)[-1] if isinstance(tag, str) else ""


def is_cite_xref(x):
    t = x.get("ref-type")
    if t:
        return t in CITE_XREFS
    return bool(re.fullmatch(r"[\d\s,–—\-]+", "".join(x.itertext()).strip() or "x"))


def math_text(e):
    parts = []

    def w(x):
        if ln(x.tag) in MATH_SKIP:
            return
        if x.text:
            parts.append(x.text)
        for c in x:
            w(c)
            if c.tail:
                parts.append(c.tail)

    w(e)
    return "".join(parts)


def flat(e, buf):
    """Append the text of `e`'s content to buf, flattening inline markup."""
    if e.text:
        buf.append(e.text)
    for c in e:
        tag = ln(c.tag)
        if tag in DROP:
            pass
        elif tag == "xref":
            if is_cite_xref(c):
                buf.append(SENT)
            else:
                flat(c, buf)
        elif tag in MATH:
            # A multi-row formula (matrix, piecewise definition) flattens to
            # run-together symbols; leave a space instead.
            if any(ln(d.tag) == "mtable" for d in c.iter()):
                buf.append(" ")
            else:
                buf.append(math_text(c))
        elif tag in BLOCKISH:
            buf.append(" ")
            flat(c, buf)
            buf.append(" ")
        else:
            flat(c, buf)
        if c.tail:
            buf.append(c.tail)


def tidy(s):
    s = _RE_INVIS.sub("", s)
    if SENT in s:
        s = _RE_MERGE.sub(SENT, s)
        # A bracket group that held only citations collapses to one marker, so
        # an enclosing "([12])" or "(see [12])" is emptied on the next pass.
        for _ in range(4):
            s2 = _RE_MERGE.sub(SENT, _RE_BRCONN.sub(SENT, _RE_BRCHAIN.sub(SENT, s)))
            if s2 == s:
                break
            s = s2
        s = _RE_LEAD.sub("", s)
        s = _RE_TRAIL.sub("", s)
        s = _RE_MID.sub("", s)
        s = re.sub(r"\s*" + SENT, "", s)
        s = _RE_WS.sub(" ", s)
        s = re.sub(r"\s+([.,)\]])", r"\1", s)
        s = re.sub(r"([\(\[])\s+", r"\1", s)
    return _RE_WS.sub(" ", s).strip()


def text_of(e):
    buf = []
    flat(e, buf)
    return tidy("".join(buf))


def sec_title(sec):
    t = sec.find("title")
    return text_of(t) if t is not None else ""


def is_back(*names):
    return any(k in n.lower().replace("-", " ").replace("_", " ") for n in names if n for k in BACKMATTER)


def collect(node, out):
    """Append the paragraph texts under `node` (itself included) to `out` in
    document order; nested secs are folded in, back-matter-style ones skipped."""
    tag = ln(node.tag)
    if tag in DROP:
        return
    if tag == "p":
        t = text_of(node)
        # Under 4 words is what is left of a paragraph once its display
        # equation is gone ("Where,", "And,"), not prose.
        if len(t.split()) >= MIN_PARA_WORDS:
            out.append(t)
    elif tag == "sec" and is_back(sec_title(node), node.get("sec-type", "")):
        return
    else:
        for c in node:
            collect(c, out)


def sec_paras(sec):
    out = []
    for c in sec:
        collect(c, out)
    return out


def n_words(paras):
    return sum(len(p.split()) for p in paras)


ROLE_RULES = (
    ("introduction", ("intro", "background")),
    ("methods", ("method", "materials", "experimental", "patients and", "study design", "procedure")),
    ("results", ("result", "findings")),
    ("discussion", ("discussion",)),
    ("conclusion", ("conclusion", "concluding")),
)


def classify(label):
    h = label.lower().replace("-", " ").replace("_", " ").replace("|", " ")
    if "result" in h and "discussion" in h:
        return "results_discussion"
    for role, keys in ROLE_RULES:
        if any(k in h for k in keys):
            return role
    return "other"


def role_of(heading, sec_type):
    r = classify(heading)
    return r if r != "other" else classify(sec_type)


def split_parts(paras):
    """Consecutive runs of paragraphs of at most PART_MAX words (a paragraph
    longer than that stays whole)."""
    parts, cur, cur_w = [], [], 0
    for p in paras:
        w = len(p.split())
        if cur and cur_w + w > PART_MAX:
            parts.append(cur)
            cur, cur_w = [], 0
        cur.append(p)
        cur_w += w
    if cur:
        parts.append(cur)
    return parts


def split_of(pmcid):
    b = int(hashlib.sha1(pmcid.encode()).hexdigest(), 16) % 100
    return "train" if b < 88 else "dev" if b < 92 else "test"


def load_xml(raw):
    try:
        return ET.fromstring(raw)
    except ET.ParseError:
        # Undefined named entities (&nbsp; etc.): map them through html.entities.
        def fix(m):
            cp = html.entities.name2codepoint.get(m.group(1))
            return f"&#{cp};" if cp else ""
        raw2 = re.sub(rb"&([A-Za-z][A-Za-z0-9]*);",
                      lambda m: fix(m).encode() if m.group(1) not in
                      (b"amp", b"lt", b"gt", b"quot", b"apos") else m.group(0), raw)
        return ET.fromstring(raw2)


_CC_RE = re.compile(r"creativecommons\.org/(?:licenses/([a-z]+(?:-[a-z]+)*)|publicdomain/([a-z]+))", re.IGNORECASE)


def license_reason(root):
    """None if the article's own <license> block is CC BY or CC0, else a drop reason."""
    am = root.find("front/article-meta")
    if am is None:
        am = root.find(".//article-meta")
    blocks = am.findall("permissions/license") if am is not None else []
    if not blocks:
        return "license_missing"
    kinds = set()
    for b in blocks:
        s = ET.tostring(b, encoding="unicode")
        for m in _CC_RE.finditer(s):
            kinds.add(m.group(1).lower() if m.group(1) else "pd-" + m.group(2).lower())
    if not kinds:
        return "license_no_cc_url"
    bad = sorted(k for k in kinds if k not in ("by", "pd-zero"))
    return f"license_{bad[0]}" if bad else None


def article_rows(pmcid, meta, root, counters):
    """Rows for one licence-passing article, or [] with counters updated."""
    body = root.find("body")
    if body is None:
        counters["no_body"] += 1
        return None
    units = []  # (heading, role, paras)
    for top in body.findall("sec"):
        h, st = sec_title(top), top.get("sec-type", "")
        if is_back(h, st):
            counters["units_backmatter"] += 1
            continue
        role = role_of(h, st)
        paras = sec_paras(top)
        kids = top.findall("sec")
        if kids and n_words(paras) > EXPAND_ABOVE:
            direct = []  # paragraphs of the top-level sec that sit outside any child sec

            def flush():
                if direct:
                    units.append((h, role, list(direct)))
                    direct.clear()

            for c in top:
                if ln(c.tag) != "sec":
                    collect(c, direct)
                    continue
                flush()
                kh, kst = sec_title(c), c.get("sec-type", "")
                if is_back(kh, kst):
                    counters["units_backmatter"] += 1
                    continue
                # A wrapper such as "Main text" says nothing about its children,
                # so an 'other' top-level role defers to the child's own.
                krole = role if role != "other" else role_of(kh, kst)
                units.append((f"{h} > {kh}" if h and kh else (kh or h), krole, sec_paras(c)))
            flush()
        else:
            units.append((h, role, paras))
    rows = []
    for heading, role, paras in units:
        w = n_words(paras)
        if w < MIN_WORDS:
            counters["units_short"] += 1
            continue
        parts = split_parts(paras) if w > SPLIT_ABOVE else [paras]
        if len(parts) > 1:
            counters["units_split"] += 1
        for i, part in enumerate(parts):
            pw = n_words(part)
            if pw < MIN_WORDS:
                counters["parts_short"] += 1
                continue
            rows.append(dict(sid=f"{pmcid}:{len(rows)}", pmcid=pmcid, journal=meta.get("journal", ""),
                             year=meta.get("year"), article_title=meta.get("title", ""),
                             heading=heading, role=role, part=i, n_words=pw,
                             text="\n".join(part), split=split_of(pmcid)))
    return rows


def quantiles(xs, qs=(0, 0.05, 0.25, 0.5, 0.75, 0.95, 1)):
    if not xs:
        return None
    xs = sorted(xs)
    out = {}
    for q in qs:
        pos = q * (len(xs) - 1)
        lo = int(pos)
        hi = min(lo + 1, len(xs) - 1)
        out[f"p{int(q * 100)}"] = round(xs[lo] + (xs[hi] - xs[lo]) * (pos - lo), 1)
    return out


def cmd_parse(a):
    arts = read_jsonl(ARTICLES)
    if not arts:
        sys.exit("parse: data/articles.jsonl is missing or empty; run `sample` first")
    metas = {}
    for r in arts:
        metas.setdefault(r["pmcid"], r)
    counters = Counter()
    drops = Counter()
    kept = 0
    rows_per_article, words, by_role, by_split = [], [], Counter(), Counter()
    art_by_split = Counter()
    journal_rows, journal_arts = Counter(), Counter()
    OUT.mkdir(parents=True, exist_ok=True)
    tmp = SECTIONS.with_name(SECTIONS.name + ".part")
    with gzip.open(tmp, "wt", encoding="utf-8", compresslevel=6) as fout:
        for pmcid, meta in metas.items():
            p = xml_path(pmcid)
            if not p.exists():
                drops["no_xml"] += 1
                continue
            try:
                with gzip.open(p, "rb") as f:
                    root = load_xml(f.read())
            except (ET.ParseError, OSError, EOFError):
                drops["xml_parse_error"] += 1
                continue
            reason = license_reason(root)
            if reason:
                drops[reason] += 1
                continue
            rows = article_rows(pmcid, meta, root, counters)
            if rows is None:
                drops["no_body"] += 1
                continue
            if not rows:
                drops["no_units"] += 1
                continue
            kept += 1
            rows_per_article.append(len(rows))
            art_by_split[rows[0]["split"]] += 1
            journal_arts[meta.get("journal", "")] += 1
            for r in rows:
                fout.write(json.dumps(r, ensure_ascii=False) + "\n")
                words.append(r["n_words"])
                by_role[r["role"]] += 1
                by_split[r["split"]] += 1
                journal_rows[meta.get("journal", "")] += 1
    os.replace(tmp, SECTIONS)
    listed = len(metas)
    lic = sum(v for k, v in drops.items() if k.startswith("license_"))
    n_xml = listed - drops["no_xml"]
    stats = dict(
        articles=dict(listed=listed, with_xml=n_xml, kept=kept, dropped=dict(sorted(drops.items())),
                      dropped_for_licence=lic,
                      licence_drop_share_of_xml=round(lic / n_xml, 4) if n_xml else None),
        units=dict(rows=len(words), skipped_backmatter=counters["units_backmatter"],
                   dropped_short=counters["units_short"], units_split=counters["units_split"],
                   rows_over_700_words=sum(1 for w in words if w > SPLIT_ABOVE),
                   tail_parts_dropped_short=counters["parts_short"]),
        sections_per_article=dict(mean=round(sum(rows_per_article) / len(rows_per_article), 2)
                                  if rows_per_article else None,
                                  **(quantiles(rows_per_article) or {})),
        n_words=quantiles(words),
        total_words=sum(words),
        rows_by_role=dict(by_role.most_common()),
        rows_by_split=dict(by_split.most_common()),
        articles_by_split=dict(art_by_split.most_common()),
        top_journals=[dict(journal=j, articles=n, rows=journal_rows[j])
                      for j, n in journal_arts.most_common(20)],
    )
    DATA.mkdir(parents=True, exist_ok=True)
    tmp = STATS.with_name(STATS.name + ".part")
    tmp.write_text(json.dumps(stats, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, STATS)
    say(f"parse: {listed} listed, {kept} kept, {len(words)} rows; drops {dict(drops)}")
    say("done -> out/sections.jsonl.gz, data/corpus_stats.json")


# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sample", help="draw articles by random first-publication date")
    s.add_argument("--n-articles", type=int, required=True)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--workers", type=int, default=4, help="dates queried in parallel (<=5 req/s total)")
    s.set_defaults(fn=cmd_sample)
    f = sub.add_parser("fetch", help="download JATS XML for data/articles.jsonl")
    f.add_argument("--workers", type=int, default=6)
    f.add_argument("--source", choices=("epmc", "ncbi"), default=None,
                   help=f"default {DEFAULT_SOURCE}")
    f.add_argument("--batch", type=int, default=20, help="ids per efetch call (ncbi source)")
    f.add_argument("--limit", type=int, default=0, help="only the first N articles (testing)")
    f.set_defaults(fn=cmd_fetch)
    p = sub.add_parser("parse", help="XML -> section units")
    p.set_defaults(fn=cmd_parse)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
