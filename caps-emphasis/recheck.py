"""Sub-5-minute fixture: check RESULTS.md against the artifacts it reports.

Every number quoted in the prose is re-read from the JSON and compared. Run this
after editing either side; a drift between writeup and data fails here rather
than surviving into a published post.
"""
import json, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
FAIL = []


def load(name):
    p = os.path.join(HERE, name)
    if not os.path.exists(p):
        FAIL.append(f"missing artifact: {name}")
        return None
    return json.load(open(p))


def check(label, claimed, actual, tol=0.005):
    if actual is None:
        FAIL.append(f"{label}: no value in artifact")
        return
    if abs(claimed - actual) > tol:
        FAIL.append(f"{label}: prose says {claimed}, artifact says {actual:.4f}")
    else:
        print(f"  ok  {label:44} {actual:+.4f}")


def prose_numbers(pattern, text, group=1):
    m = re.search(pattern, text)
    return float(m.group(group)) if m else None


def main():
    md_path = os.path.join(HERE, "RESULTS.md")
    if not os.path.exists(md_path):
        print("RESULTS.md not written yet — nothing to check")
        return 0
    md = open(md_path).read()
    sup = load("suppression.json")
    dose = load("dose.json")

    if sup:
        n = sup["n_items"]
        claimed_n = prose_numbers(r"(\d+)\s+items", md)
        if claimed_n is not None and int(claimed_n) != n:
            FAIL.append(f"item count: prose {int(claimed_n)}, artifact {n}")
        else:
            print(f"  ok  item count {n}")

        # every table row of the form | keyword | +N | x.xxx | must match
        for m in re.finditer(
                r"^\|\s*`?(\w+)`?\s*\|\s*\+?(\d+)\s*\|\s*([-+]?\d+\.\d+)\s*\|", md, re.M):
            key, dtok, val = m.group(1), int(m.group(2)), float(m.group(3))
            row = next((e for e in sup["caps_effects"] if e["keyword"] == key), None)
            if row is None:
                continue
            if row["caps_extra_tokens"] != dtok:
                FAIL.append(f"{key}: prose +{dtok} tokens, artifact "
                            f"+{row['caps_extra_tokens']}")
            check(f"caps effect [{key}]", val, row["caps_effect_nats"])

    if dose:
        for r in dose["rows"]:
            pat = rf"^\|\s*{int(r['frac']*100)}%\s*\|.*?\|\s*([-+]?\d+\.\d+)\s*\|"
            m = re.search(pat, md, re.M)
            if m:
                check(f"dose {int(r['frac']*100)}%", float(m.group(1)), r["suppression"])

    if FAIL:
        print("\nFAIL:")
        for f in FAIL:
            print("  -", f)
        return 1
    print("\nrecheck passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
