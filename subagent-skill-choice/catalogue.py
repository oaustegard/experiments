#!/usr/bin/env python3
"""Snapshot the skill catalogue a model chooses from: name and description of every skill.

    python3 catalogue.py SKILLS_DIR OUT.json

Reads each <skill>/SKILL.md frontmatter the way the harness lists skills. The
built-in `workflow-authoring` skill is added by hand, because it is not in the
repo and it competes for the same requests.
"""

import json
import sys
from pathlib import Path

import yaml

WORKFLOW_AUTHORING = ("Reference for writing a Workflow tool script (script API and gotchas, resume, quality "
                      "patterns, worked examples). Load before authoring a script for a workflow the user "
                      "already opted into; it does not itself authorize running one.")


def main(src: Path, out: Path) -> None:
    cat = {}
    for f in sorted(src.glob("*/SKILL.md")):
        text = f.read_text()
        if not text.startswith("---"):
            continue
        try:
            fm = yaml.safe_load(text.split("---", 2)[1])
        except yaml.YAMLError:
            continue
        if isinstance(fm, dict) and fm.get("name") and fm.get("description"):
            cat[fm["name"]] = " ".join(str(fm["description"]).split())
    cat["workflow-authoring"] = WORKFLOW_AUTHORING
    out.write_text(json.dumps(cat, indent=1))
    print(f"{len(cat)} skills -> {out}")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
