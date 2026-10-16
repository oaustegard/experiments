"""Collect TLO documents from a tinylittleoffice checkout: train = templates/test corpus, test = samples/."""
import glob, json, os, sys
root = sys.argv[1]; out = sys.argv[2]
train, test = {}, {}
for f in sorted(glob.glob(f"{root}/test/doc/corpus/**/*.json", recursive=True)):
    d = json.load(open(f))
    for x in d.get("docs", []) if isinstance(d, dict) else []:
        t = x.get("text") if isinstance(x, dict) else None
        if t and len(t) > 20: train[t] = f"{os.path.basename(f)}:{x.get('name')}"
for f in sorted(glob.glob(f"{root}/samples/*.tlo")):
    t = open(f, encoding="utf-8").read()
    test[t] = os.path.basename(f)
for t in list(train):
    if t in test: del train[t]
json.dump({"train": [{"name": n, "text": t} for t, n in train.items()],
           "test": [{"name": n, "text": t} for t, n in test.items()]}, open(out, "w"), ensure_ascii=False, indent=0)
print(len(train), "train docs,", sum(map(len, train)), "chars;", len(test), "test docs,", sum(map(len, test)), "chars")
