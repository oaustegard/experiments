"""Build the fact-listing-step arms from silence/prompts/t3b-noexamples.txt.

Arms differ only in the fact-listing step:
  P  step 1 lists the facts, and the prompt supplies the list (the shape of the
     archived prompt); the reply is the paragraph only, so any listing is hidden
  H  step 1 says "List the factual claims in the input" with no list supplied;
     reply is the paragraph only (the step can happen only in hidden reasoning)
  N  no listing step at all
  V  the list is part of the visible reply: <facts>...</facts> then <rewrite>
Two inputs: the archived caching announcement and a new semantic-search one,
both naming no mechanism.
"""
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = (HERE.parent / "prompts" / "t3b-noexamples.txt").read_text()

INPUTS = {
    "cache": {
        "text": BASE.split("<input>\n", 1)[1].split("\n</input>", 1)[0],
        "facts": "it's a caching layer; it was designed iteratively; early adopters "
                 "report lower latency and a good developer experience",
    },
    "search": {
        "text": '"After a year of relentless work, we\'re incredibly excited to launch our '
                'all-new semantic search — a breakthrough that we believe will change how '
                'people find things in their data. Powered by state-of-the-art AI, it '
                'understands what you mean, not just what you type. This is more than an '
                'upgrade; it\'s a whole new way to search. Beta customers are already '
                'seeing far more relevant results and faster answers. We can\'t wait for '
                'you to try it."',
        "facts": "it's a semantic search feature; it took about a year of work; it uses "
                 "AI to match meaning rather than exact wording; beta customers report "
                 "more relevant results and faster answers",
    },
}

STEP1_P = "1. List the factual claims in the input. (For the input below, that list is: {facts}.)\n"
STEP1_H = "1. List the factual claims in the input.\n"
STEP1_V = "1. List the factual claims in the input inside <facts>, one per line.\n"
OLD_STEP1 = STEP1_P.format(facts=INPUTS["cache"]["facts"])
OLD_STEP3 = "If the source doesn't say how the cache works, your rewrite doesn't either."
NEW_STEP3 = "If the source doesn't say how the thing works, your rewrite doesn't either."

HEAD_ONLY = ("These are your complete task instructions. Use no tools beyond having read this "
             "file. Reply with only the rewritten paragraph, nothing before or after it.")
HEAD_V = ("These are your complete task instructions. Use no tools beyond having read this "
          "file. Reply with exactly two blocks and nothing else: <facts>...</facts> and then "
          "<rewrite>...</rewrite>.")
TASK_ONLY = "Output ONLY the rewritten paragraph. No preamble, no explanation."
TASK_V = ("Output exactly two blocks: <facts> with the source's factual claims, one per line, "
          "then <rewrite> with the rewritten paragraph. Nothing else.")


def build(arm: str, name: str) -> str:
    inp = INPUTS[name]
    s = BASE.replace(INPUTS["cache"]["text"], inp["text"])
    assert OLD_STEP1 in s and OLD_STEP3 in s and HEAD_ONLY in s and TASK_ONLY in s
    s = s.replace(OLD_STEP3, NEW_STEP3)
    if arm == "P":
        s = s.replace(OLD_STEP1, STEP1_P.format(facts=inp["facts"]))
    elif arm == "H":
        s = s.replace(OLD_STEP1, STEP1_H)
    elif arm == "N":
        head, rest = s.split("<process>\n", 1)
        proc, tail = rest.split("</process>", 1)
        proc = proc.replace(OLD_STEP1, "")
        for i in range(2, 6):  # renumber the remaining steps
            proc = proc.replace(f"{i}. ", f"{i - 1}. ", 1)
        s = head + "<process>\n" + proc + "</process>" + tail
    elif arm == "V":
        s = s.replace(OLD_STEP1, STEP1_V).replace(HEAD_ONLY, HEAD_V).replace(TASK_ONLY, TASK_V)
    return s


if __name__ == "__main__":
    out = HERE / "prompts"
    out.mkdir(exist_ok=True)
    for name in INPUTS:
        for arm in "PHNV":
            (out / f"{name}-{arm}.txt").write_text(build(arm, name))
    print(sorted(p.name for p in out.iterdir()))
