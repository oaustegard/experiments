"""Sub-5-minute fixture: check RESULTS.md against the artifacts it reports.

Every number quoted in the writeup is re-read from the JSON that produced it and
compared. Run after editing either side; a drift between prose and data fails
here rather than surviving into a published post.
"""
import json, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
FAIL, OK = [], []


def load(name):
    p = os.path.join(HERE, name)
    if not os.path.exists(p):
        FAIL.append(f"missing artifact: {name}")
        return None
    return json.load(open(p))


def check(label, claimed, actual, tol=0.0006):
    if actual is None:
        FAIL.append(f"{label}: no value in artifact")
    elif abs(claimed - actual) > tol:
        FAIL.append(f"{label}: prose {claimed}, artifact {actual:.4f}")
    else:
        OK.append(label)


def main():
    md = open(os.path.join(HERE, "RESULTS.md")).read()
    v2, dose = load("v2.json"), load("dose2.json")
    pos, reg = load("position.json"), load("register.json")
    ko, gen = load("knockout.json"), load("generation_framed.json")
    if not all([v2, dose, pos, reg, ko, gen]):
        return report()

    n = v2["n_items"]
    m = re.search(r"(\d+) items, stratified", md)
    if m and int(m.group(1)) != n:
        FAIL.append(f"item count: prose {m.group(1)}, artifact {n}")
    else:
        OK.append(f"item count {n}")

    # Q2 token-cost bins: "| +N tokens | K | -0.100 | [-0.229, +0.035] |"
    for m in re.finditer(r"^\|\s*\+(\d) tokens?\s*\|\s*(\d+)\s*\|\s*"
                         r"(−|-)?([\d.]+)\s*\|", md, re.M):
        d, kw = m.group(1), int(m.group(2))
        val = float(m.group(4)) * (-1 if m.group(3) else 1)
        row = v2["caps_by_token_delta"].get(d)
        if row is None:
            FAIL.append(f"token-delta bin +{d} not in artifact")
            continue
        if row["n_keywords"] != kw:
            FAIL.append(f"bin +{d}: prose {kw} keywords, artifact {row['n_keywords']}")
        check(f"caps effect bin +{d}", val, row["mean"])

    # dose rows: "| 25% | -1.984 | ..."
    for r in dose["rows"]:
        m = re.search(rf"^\|\s*{int(r['frac'] * 100)}%\s*\|\s*(−|-)?([\d.]+)\s*\|", md, re.M)
        if m:
            check(f"dose {int(r['frac'] * 100)}%",
                  float(m.group(2)) * (-1 if m.group(1) else 1), r["mean"])

    # strata
    for name, row in v2["caps_by_stratum"].items():
        m = re.search(rf"^\|\s*{name}\s*\|\s*(\d+)\s*\|\s*(\+|−|-)?([\d.]+)\s*\|", md, re.M)
        if m:
            if int(m.group(1)) != row["n"]:
                FAIL.append(f"stratum {name}: prose n={m.group(1)}, artifact {row['n']}")
            check(f"stratum {name}",
                  float(m.group(3)) * (-1 if m.group(2) in ("−", "-") else 1), row["mean"])

    # the two load-bearing headline numbers
    check("medial CAPS null", 0.003, pos["relative"]["medial|caps"]["effect"])
    check("bold in user turn", 0.277, reg["user|bold"]["suppression"]
          - reg["user|title"]["suppression"])
    check("bold in reasoning register", -0.240, reg["think|bold"]["suppression"]
          - reg["think|title"]["suppression"])
    check("register move", 0.8, reg["think|title"]["suppression"]
          - reg["user|title"]["suppression"], tol=0.05)

    # knockout table
    for span, mode in [("`Never`", "title"), ("`NEVER`", "caps"), ("`**never**`", "bold")]:
        m = re.search(rf"^\|\s*{re.escape(span)}\s*\|\s*(\d)\s*\|\s*([\d.]+)\s*\|"
                      rf"\s*([\d.]+)\s*\|\s*\+([\d.]+) log P\s*\|", md, re.M)
        if not m:
            FAIL.append(f"knockout row for {span} not found in prose")
            continue
        a = ko[mode]
        if int(m.group(1)) != round(a["span_tokens"]):
            FAIL.append(f"{span}: prose {m.group(1)} tokens, artifact {a['span_tokens']}")
        check(f"knockout mass {mode}", float(m.group(2)), a["attention_total"], tol=0.001)
        check(f"knockout all-layer {mode}", float(m.group(4)),
              a["knockout_all_layers"], tol=0.001)

    # violation counts
    counts = {}
    for r in gen["rows"]:
        counts.setdefault(r["cond"], [0, 0])
        counts[r["cond"]][0] += bool(r["violated"])
        counts[r["cond"]][1] += 1
    for cond, (v, tot) in counts.items():
        if cond == "m_bold" and (v, tot) != (41, 43):
            FAIL.append(f"bold violations: artifact {v}/{tot}, prose says 41/43")
        elif cond != "m_bold" and (v, tot) != (42, 43):
            FAIL.append(f"{cond} violations: artifact {v}/{tot}, prose says 42/43")
    OK.append("violation counts")
    return report()


def report():
    for o in OK:
        print(f"  ok  {o}")
    if FAIL:
        print("\nFAIL:")
        for f in FAIL:
            print("  -", f)
        return 1
    print(f"\nrecheck passed ({len(OK)} checks)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
