#!/usr/bin/env python3
"""Generate seeded SQL scripts (sql/gen_<kind>_<nn>.sql) that exercise SQLite's number
formatting, arithmetic, functions, dates, JSON, aggregates and sorting with many
deterministic cases per script.  The output is committed; this is the source of record.

    python3 tests/scripts/gen_sqlite_generated.py          # 8 kinds x 40 scripts
"""
import glob
import os
import random

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(HERE), "suites", "sqlite", "sql")
PER_KIND = 40

INTS = ["0", "1", "-1", "2", "7", "-7", "10", "255", "256", "1000", "65535", "2147483647", "-2147483648", "4294967296",
        "9007199254740993", "9223372036854775807", "-9223372036854775807", "123456789012"]
REALS = ["0.0", "0.5", "-0.5", "1.5", "2.5", "3.14159265358979", "1e10", "1e-10", "123456.789", "-0.000123", "1e100",
         "9.99999999999999e22", "4.35", "0.1", "0.7", "1.0000000000000002", "5e-324", "1.7976931348623157e308", "100.0"]
TEXTS = ["'abc'", "''", "'12'", "'1.5e2'", "' 7 '", "'0x1F'", "'-3abc'", "'Hello World'", "'ÀÉÎ'", "'a,b,c'", "'x'", "'NaN'", "'1e999'"]
BIN = ["+", "-", "*", "/", "%", "&", "|", "<<", ">>", "||", "<", "<=", "=", "!=", "IS", "IS NOT"]
FN1 = ["abs", "round", "ceil", "floor", "sign", "typeof", "length", "upper", "lower", "hex", "quote", "trunc", "sqrt", "exp", "ln", "sin", "cos", "atan", "unicode", "total"]
FN2 = ["max", "min", "coalesce", "nullif", "ifnull", "round", "instr", "substr", "printf", "pow", "mod", "atan2", "log"]


def val(r):
    k = r.random()
    if k < 0.45:
        return r.choice(INTS)
    if k < 0.8:
        return r.choice(REALS)
    return r.choice(TEXTS)


def expr(r, depth=0):
    k = r.random()
    if depth > 1 or k < 0.3:
        return val(r)
    if k < 0.65:
        return "(%s %s %s)" % (expr(r, depth + 1), r.choice(BIN), expr(r, depth + 1))
    if k < 0.85:
        return "%s(%s)" % (r.choice(FN1), expr(r, depth + 1))
    return "%s(%s, %s)" % (r.choice(FN2), expr(r, depth + 1), expr(r, depth + 1))


def gen_arith(r, i):
    return "\n".join("SELECT %s, typeof(%s);" % ((e,) * 2) for e in (expr(r) for _ in range(40)))


def gen_format(r, i):
    specs = ["%d", "%5d", "%-8d|", "%05d", "%+d", "%x", "%X", "%o", "%c", "%s", "%10s", "%-10s|", "%.3s", "%q", "%Q", "%w",
             "%f", "%.0f", "%.3f", "%10.2f", "%e", "%.2e", "%g", "%.10g", "%G", "%#x", "%,d", "%!.2f", "%lld", "%i", "%u"]
    lines = []
    for _ in range(40):
        a = r.choice(INTS + REALS + TEXTS)
        lines.append("SELECT printf('%s', %s);" % (r.choice(specs), a))
        lines.append("SELECT format('[%s]', %s), format('%s %s', %s, %s);" % (r.choice(specs), val(r), r.choice(specs), r.choice(specs), val(r), val(r)))
    return "\n".join(lines)


def gen_cast(r, i):
    types = ["INTEGER", "REAL", "TEXT", "BLOB", "NUMERIC"]
    return "\n".join("SELECT CAST(%s AS %s), typeof(CAST(%s AS %s)), %s + 0, %s * 1.0;" % (v, r.choice(types), v, r.choice(types), v, v)
                     for v in (val(r) for _ in range(40)))


def gen_datetime(r, i):
    mods = ["+1 day", "-1 month", "+1 year", "start of month", "start of year", "start of day", "weekday 3", "+36 hours", "-90 minutes",
            "+0.5 days", "unixepoch", "localtime", "utc", "ceiling", "floor", "subsec", "+10 seconds", "-100 years", "+1000 days"]
    lines = []
    for _ in range(40):
        y, m, d = r.randint(1, 9999), r.randint(1, 12), r.randint(1, 31)
        ts = "%04d-%02d-%02d %02d:%02d:%02d" % (y, m, d, r.randint(0, 24), r.randint(0, 61), r.randint(0, 61))
        ms = ", ".join("'%s'" % x for x in r.sample(mods, r.randint(0, 3)))
        lines.append("SELECT datetime('%s'%s), julianday('%s'), strftime('%%Y-%%m-%%d %%w %%j %%W %%s', '%s'%s);" % (
            ts, (", " + ms) if ms else "", ts, ts, (", " + ms) if ms else ""))
    return "\n".join(lines)


def js(r, depth=0):
    k = r.random()
    if depth > 2 or k < 0.4:
        return r.choice(["1", "-2.5", "1e3", "true", "false", "null", '"s"', '"é\\n"', "0", "123456789012345"])
    if k < 0.7:
        return "[" + ",".join(js(r, depth + 1) for _ in range(r.randint(0, 4))) + "]"
    return "{" + ",".join('"k%d":%s' % (j, js(r, depth + 1)) for j in range(r.randint(0, 4))) + "}"


def gen_json(r, i):
    lines = []
    for _ in range(25):
        j = js(r)
        lines.append("SELECT json('%s'), json_valid('%s'), json_type('%s'), json_array_length('%s'), json_extract('%s', '$.k0'), json_extract('%s', '$[1]');" % ((j,) * 6))
        lines.append("SELECT json_set('%s', '$.k1', 5), json_remove('%s', '$[0]'), json_quote('%s'), (SELECT count(*) FROM json_tree('%s'));" % ((j,) * 4))
    return "\n".join(lines)


def gen_aggregate(r, i):
    n = r.randint(20, 120)
    rows = ",".join("(%d,%s,%s,%s)" % (j, r.choice(["'a'", "'b'", "'c'", "NULL"]), r.choice(INTS + ["NULL"]), r.choice(REALS + ["NULL"])) for j in range(n))
    return ("CREATE TABLE t(id INTEGER PRIMARY KEY, g, i, r);\nINSERT INTO t VALUES%s;\n" % rows +
            "SELECT g, count(*), count(i), sum(i), total(i), avg(i), min(i), max(i), sum(r), avg(r), group_concat(i) FROM t GROUP BY g ORDER BY g;\n"
            "SELECT g, i, r, row_number() OVER (PARTITION BY g ORDER BY i, id), sum(r) OVER (ORDER BY id ROWS 3 PRECEDING), "
            "rank() OVER (ORDER BY i), lag(i) OVER (ORDER BY id), ntile(3) OVER (ORDER BY id) FROM t ORDER BY id;\n"
            "SELECT * FROM t WHERE i > 5 OR r < 1.0 ORDER BY r, i, id LIMIT 20;\nSELECT i, r, i + r, i * r, i / (r + 1) FROM t ORDER BY id;")


def gen_sort(r, i):
    pool = INTS + REALS + TEXTS + ["NULL", "x'00'", "x'ff01'", "''", "'B'", "'b'", "' '", "'é'"]
    vals = ",".join("(%s,%s)" % (r.choice(pool), r.choice(pool)) for _ in range(60))
    return ("CREATE TABLE t(a, b);\nINSERT INTO t VALUES%s;\n" % vals +
            "SELECT a, b, typeof(a) FROM t ORDER BY a, b;\nSELECT a, b FROM t ORDER BY b DESC, a;\n"
            "SELECT a FROM t ORDER BY a COLLATE NOCASE, b;\nSELECT DISTINCT a FROM t ORDER BY 1;\n"
            "SELECT a, count(*) FROM t GROUP BY a ORDER BY 2 DESC, 1;\nSELECT min(a), max(a), min(b), max(b) FROM t;\n"
            "SELECT a, b, a < b, a = b, a IS b FROM t ORDER BY rowid;")


def gen_strings(r, i):
    words = ["alpha", "Beta", "gamma delta", "épsilon", "ζ", "a%b_c", "100%", "", " pad ", "x,y,z", "tab\tchar", "q'uote", "日本語", "😀 smile"]
    lines = []
    for _ in range(35):
        s, t = "'%s'" % r.choice(words).replace("'", "''"), "'%s'" % r.choice(words).replace("'", "''")
        lines.append("SELECT %s LIKE %s, %s GLOB %s, instr(%s, %s), replace(%s, %s, '#'), substr(%s, %d, %d), trim(%s), upper(%s), lower(%s), length(%s), hex(%s), %s || %s, %s COLLATE NOCASE = %s;" % (
            s, t, s, t, s, t, s, t, s, r.randint(-5, 6), r.randint(-3, 8), s, s, s, s, s, s, t, s, t))
    return "\n".join(lines)


KINDS = [("arith", gen_arith), ("format", gen_format), ("cast", gen_cast), ("datetime", gen_datetime), ("json", gen_json),
         ("aggregate", gen_aggregate), ("sort", gen_sort), ("strings", gen_strings)]


def main():
    os.makedirs(OUT, exist_ok=True)
    for old in glob.glob(os.path.join(OUT, "gen_*.sql")):
        os.remove(old)
    n = 0
    for name, fn in KINDS:
        for i in range(1, PER_KIND + 1):
            r = random.Random("%s-%d" % (name, i))
            with open(os.path.join(OUT, "gen_%s_%02d.sql" % (name, i)), "w") as f:
                f.write(fn(r, i).strip("\n") + "\n")
            n += 1
    print("wrote %d generated scripts to %s" % (n, OUT))


if __name__ == "__main__":
    main()
