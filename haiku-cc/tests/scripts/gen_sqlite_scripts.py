#!/usr/bin/env python3
"""Write tests/suites/sqlite/sql/<category>_<nn>_<slug>.sql from sqlite_scripts_{a,b}.py.

The generated .sql files are committed (they are the tests); this generator is the
source of record.  Run: python3 tests/scripts/gen_sqlite_scripts.py
"""
import glob
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sqlite_scripts_a as A  # noqa: E402
import sqlite_scripts_b as B  # noqa: E402
import sqlite_scripts_c as Cc  # noqa: E402

OUT = os.path.join(os.path.dirname(HERE), "suites", "sqlite", "sql")


def main():
    os.makedirs(OUT, exist_ok=True)
    for old in glob.glob(os.path.join(OUT, "*.sql")):
        if not os.path.basename(old).startswith("gen_"):  # gen_*.sql come from gen_sqlite_generated.py
            os.remove(old)
    n = 0
    for mod in (A, B, Cc):
        for cat, items in mod.SCRIPTS.items():
            for i, (slug, sql) in enumerate(items, 1):
                path = os.path.join(OUT, "%s_%02d_%s.sql" % (cat, i, slug))
                with open(path, "w") as f:
                    f.write(sql.strip("\n") + "\n")
                n += 1
    ext = os.path.join(os.path.dirname(OUT), "sql-ext")
    os.makedirs(ext, exist_ok=True)
    for old in glob.glob(os.path.join(ext, "*.sql")):
        os.remove(old)
    for cat, items in Cc.SCRIPTS_EXT.items():
        for i, (slug, sql) in enumerate(items, 1):
            with open(os.path.join(ext, "%s_%02d_%s.sql" % (cat, i, slug)), "w") as f:
                f.write(sql.strip("\n") + "\n")
            n += 1
    print("wrote %d scripts to %s (+ sql-ext)" % (n, OUT))


if __name__ == "__main__":
    main()
