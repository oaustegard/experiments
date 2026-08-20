#!/usr/bin/env python3
"""Self-contained assertions for `extract_params.py`. No data, no network.

These pin the behaviours that are easy to regress while widening a pattern:
the English-prose guards (`and/or` is not a path, `non-zero` is not an
identifier, `don't` does not open a quote) and the span contract (every
returned value must be the exact substring its span names).

    python3 test_extract.py
"""

from extract_params import extract, values


def kinds(text):
    return {k: [v["value"] for v in spans] for k, spans in extract(text).items()}


def check(text, expect_kind, expect_value, present=True):
    got = kinds(text).get(expect_kind, [])
    hit = expect_value in got
    assert hit == present, f"{text!r}: {expect_kind}={got}, wanted {expect_value!r} present={present}"


def main():
    # spans must be exact
    for text in [
        "Find *.mov under /mnt/raid owned by user 'abc' on port 8080 at 10.0.0.1",
        "delete files older than 30 days larger than 100MB in ~/junk/logs",
        "kill the nginx process, send SIGHUP, checkout origin/main",
    ]:
        for kind, spans in extract(text).items():
            for sp in spans:
                assert text[sp["start"]:sp["end"]] == sp["value"], (kind, sp, text)

    check("Locate all *.mov files", "glob", "*.mov")
    check("Find *foo* anywhere", "glob", "*foo*")
    check("Search the /usr/src tree", "path", "/usr/src")
    check("Unzip path/to/test/file.gz", "path", "path/to/test/file.gz")
    check("Copy xyz.c to every folder", "filename", "xyz.c")
    check("Search all .VER files", "extension", ".VER")
    check("Open port 8080", "port", "8080")
    check("files bigger than 10KB", "size", "10")
    check("modified in the last 30 days", "duration", "30")
    check("connect to 192.168.1.1", "ip", "192.168.1.1")
    check("fetch https://example.com/a?b=1", "url", "https://example.com/a?b=1")
    check("ping google.com", "hostname", "google.com")
    check("broken symlinks under $path", "var", "$path")
    check("send SIGHUP to it", "signal", "HUP")
    check("run the clean_all target", "identifier", "clean_all")
    check("with 777 permission", "perm", "777")
    check("kill pid 4821", "pid", "4821")
    check("Replace with 'longer string' now", "literal", "longer string")

    # prose that owns a structural character must NOT be extracted
    check("read/write access for and/or cases", "path", "and/or", present=False)
    check("read/write access", "path", "read/write", present=False)
    check("Find directories with non-zero count", "identifier", "non-zero", present=False)
    check("List all read-only files", "identifier", "read-only", present=False)
    assert "don" not in values("jobs that don't contain anything")
    assert "e.g" not in values("some files, e.g. logs")
    assert "7*24" not in values("uptime of 7*24 hours")

    # a value repeated in the request keeps both spans
    two = extract("copy /tmp/a to /tmp/a")["path"]
    assert len(two) == 2 and two[0]["start"] != two[1]["start"], two

    print("all assertions passed")


if __name__ == "__main__":
    main()
