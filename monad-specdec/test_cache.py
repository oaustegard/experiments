"""Does CachedRunner's crop-and-reuse return the same logits as a fresh run?"""
import torch
from specdec import CachedRunner

torch.set_num_threads(4)
r = CachedRunner("PleIAs/Baguettotron")
ids = r.tok("The Roman aqueducts were built to carry water into cities.",
            add_special_tokens=False).input_ids

# Warm the cache over a longer speculative-style sequence, then crop back.
long_ids = ids + [100, 200, 300, 400]
r.logits_for(long_ids)
cropped, off = r.logits_for(ids + [100], need_from=len(ids) - 1)
got = cropped[len(ids) - 1 - off].argmax(), cropped[-1].argmax()

r2 = CachedRunner("PleIAs/Baguettotron")
fresh, off2 = r2.logits_for(ids + [100], need_from=0)
want = fresh[len(ids) - 1 - off2].argmax(), fresh[-1].argmax()

print("cropped argmax:", [int(x) for x in got])
print("fresh   argmax:", [int(x) for x in want])
print("MATCH" if got == want else "MISMATCH -> crop corrupts the cache")

maxdiff = (cropped[len(ids) - 1 - off] - fresh[len(ids) - 1 - off2]).abs().max()
print("max logit delta at shared position:", float(maxdiff))
