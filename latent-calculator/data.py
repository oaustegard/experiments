"""Synthetic arithmetic prompt/answer data for the latent-calculator experiment.

Every row is a prompt whose ANSWER follows immediately as " {answer}\\n", so the
query position t is simply the index of the last prompt token.

LENGTH SPLIT (deliberate, not a naive holdout): training uses operand digit
lengths {1,2,3,4,6} and the held-out split is length 5 -- an *interpolation*
holdout.  Slot-wise digit heads cannot extrapolate into slots that were never
active during training, so holding out the largest length would only measure
"slot 5 never trained".  Holding out 5 keeps every slot active in training and
asks whether the representation generalises across lengths.
"""

import argparse
import json
import os
import random

OPS = ["add", "sub", "mul", "cmp"]

# 12 templates; three per operator.  Each ends so that " {answer}\n" continues it.
TEMPLATES = [
    ("add", "{a} + {b} ="),
    ("add", "What is {a} plus {b}?"),
    ("add", "Add {a} and {b}:"),
    ("sub", "{a} - {b} ="),
    ("sub", "Subtract {b} from {a}."),
    ("sub", "What is {a} minus {b}?"),
    ("mul", "{a} * {b} ="),
    ("mul", "Compute {a} times {b}."),
    ("mul", "What is {a} multiplied by {b}?"),
    ("cmp", "Which is larger, {a} or {b}?"),
    ("cmp", "Compare {a} and {b}:"),
    ("cmp", "Is {a} greater or less than {b}?"),
]

TRAIN_LENGTHS = (1, 2, 3, 4, 6)
HELDOUT_LENGTH = 5
ALL_LENGTHS = (1, 2, 3, 4, 5, 6)

N_OPERAND_SLOTS = 6          # right-aligned digit slots per operand
N_RESULT_SLOTS = 12          # right-aligned digit slots for the result
BLANK = 10                   # digit class 10 == BLANK
N_DIGIT_CLASSES = 11         # 0-9 plus BLANK
KINDS = ["numeric", "greater", "less", "equal"]
CMP_WORDS = {"greater": "greater", "less": "less", "equal": "equal"}

MAX_TOKENS = 48
DEFAULT_SEED = 20260906


def sample_value(rng, length):
    if length == 1:
        return rng.randint(0, 9)
    return rng.randint(10 ** (length - 1), 10 ** length - 1)


def compute(op, a, b):
    """Reference arithmetic.  Returns (result_string, kind)."""
    if op == "add":
        return str(a + b), "numeric"
    if op == "sub":
        return str(a - b), "numeric"
    if op == "mul":
        return str(a * b), "numeric"
    if op == "cmp":
        k = "greater" if a > b else ("less" if a < b else "equal")
        return k, k
    raise ValueError(op)


def digits_right_aligned(n_str, n_slots):
    """Digit-class list, slot 0 == units.  Non-digit chars ('-') are ignored."""
    ds = [c for c in n_str if c.isdigit()]
    slots = [BLANK] * n_slots
    for i, c in enumerate(reversed(ds)):
        if i >= n_slots:
            raise ValueError(f"{n_str} does not fit in {n_slots} slots")
        slots[i] = int(c)
    return slots


def query_target(op, a, b):
    """Labels for the query head: op index + 6 slots for a + 6 slots for b."""
    return {
        "op": OPS.index(op),
        "a_slots": digits_right_aligned(str(a), N_OPERAND_SLOTS),
        "b_slots": digits_right_aligned(str(b), N_OPERAND_SLOTS),
    }


def result_target(result_string):
    """Labels for the result encoder: sign(2) + kind(4) + 12 digit slots."""
    if result_string in CMP_WORDS:
        return {"sign": 0, "kind": KINDS.index(result_string),
                "slots": [BLANK] * N_RESULT_SLOTS}
    neg = result_string.startswith("-")
    return {"sign": 1 if neg else 0, "kind": 0,
            "slots": digits_right_aligned(result_string, N_RESULT_SLOTS)}


def gen_rows(n, lengths, rng, seen, require_len5=False, equal_rate=0.05):
    rows = []
    guard = 0
    while len(rows) < n:
        guard += 1
        if guard > 200 * n + 10000:
            raise RuntimeError("could not generate enough unique rows")
        op = OPS[rng.randrange(len(OPS))]
        if require_len5:
            la, lb = rng.choice(ALL_LENGTHS), rng.choice(ALL_LENGTHS)
            if rng.random() < 0.5:
                la = HELDOUT_LENGTH
            else:
                lb = HELDOUT_LENGTH
            if la != HELDOUT_LENGTH and lb != HELDOUT_LENGTH:
                continue
        else:
            la, lb = rng.choice(lengths), rng.choice(lengths)
        a, b = sample_value(rng, la), sample_value(rng, lb)
        if op == "cmp" and rng.random() < equal_rate:
            b = a
            lb = la
        key = (op, a, b)
        if key in seen:
            continue
        seen.add(key)
        tmpl = [t for t in TEMPLATES if t[0] == op][rng.randrange(3)][1]
        prompt = tmpl.format(a=a, b=b)
        result_string, _kind = compute(op, a, b)
        rows.append({
            "prompt": prompt,
            "answer": " " + result_string + "\n",
            "op": op,
            "a": a,
            "b": b,
            "result_string": result_string,
            "lengths": [len(str(a)), len(str(b))],
        })
    return rows


def build(seed=DEFAULT_SEED, n_train=20000, n_val=2000, n_test=2000):
    rng = random.Random(seed)
    seen = set()
    splits = {}
    splits["train"] = gen_rows(n_train, TRAIN_LENGTHS, rng, seen)
    splits["val"] = gen_rows(n_val, TRAIN_LENGTHS, rng, seen)
    splits["test_in"] = gen_rows(n_test, TRAIN_LENGTHS, rng, seen)
    splits["test_len5"] = gen_rows(n_test, ALL_LENGTHS, rng, seen, require_len5=True)
    return splits


def data_dir():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def split_path(split):
    return os.path.join(data_dir(), f"{split}.jsonl")


def load_split(split, n=None):
    rows = []
    with open(split_path(split)) as f:
        for line in f:
            rows.append(json.loads(line))
            if n is not None and len(rows) >= n:
                break
    return rows


def check_token_lengths(rows, sample=400):
    """Assert prompt+answer stays under MAX_TOKENS for BOTH tokenizers."""
    from transformers import AutoTokenizer
    worst = {}
    for name in ("PleIAs/Monad", "HuggingFaceTB/SmolLM2-135M"):
        tok = AutoTokenizer.from_pretrained(name)
        step = max(1, len(rows) // sample)
        mx = 0
        for r in rows[::step]:
            n = len(tok(r["prompt"] + r["answer"])["input_ids"])
            mx = max(mx, n)
        assert mx < MAX_TOKENS, f"{name}: {mx} tokens >= {MAX_TOKENS}"
        worst[name] = mx
    return worst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--n-train", type=int, default=20000)
    ap.add_argument("--n-val", type=int, default=2000)
    ap.add_argument("--n-test", type=int, default=2000)
    ap.add_argument("--no-token-check", action="store_true")
    args = ap.parse_args()
    os.makedirs(data_dir(), exist_ok=True)
    splits = build(args.seed, args.n_train, args.n_val, args.n_test)
    for name, rows in splits.items():
        with open(split_path(name), "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        print(f"{name}: {len(rows)} rows -> {split_path(name)}")
    if not args.no_token_check:
        allrows = [r for rs in splits.values() for r in rs]
        print("max prompt+answer tokens:", check_token_lengths(allrows))


if __name__ == "__main__":
    main()
