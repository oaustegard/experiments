#!/usr/bin/env python3
"""Expert pruning tools for Inkling 2 Mini GGUFs (inkling2_loop layout).

  keep   usage.tsv[,more.tsv] K out.json  pick the K most-used experts per physical layer
  mask   model.gguf keep.json          in place: push pruned experts' selection bias to -1e9
  unmask model.gguf backup.npz         restore the biases saved by `mask`
  slice  model.gguf keep.json out.gguf write a smaller GGUF holding only the kept experts

usage.tsv is written by the patched llama-perplexity (LLAMA_EXPERT_USAGE): one line
per layer slot, `slot<TAB>count_0<TAB>...`. Slots past the physical layer count are
later passes through the loop core and are folded onto their physical layer.
"""
import json
import sys

import numpy as np

N_PHYS, CORE_START, CORE_END = 24, 6, 18


def phys(slot):
    return slot if slot < N_PHYS else CORE_START + (slot - N_PHYS) % (CORE_END - CORE_START)


def load_usage(path, n_expert=128):
    u = np.zeros((N_PHYS, n_expert), dtype=np.int64)
    for line in open(path):
        parts = line.split('\t')
        c = np.array(parts[1:], dtype=np.int64)
        u[phys(int(parts[0])), :len(c)] += c
    return u


def cmd_keep(usage, k, out):
    # several usage files (comma separated) are blended with equal weight per file and layer
    us = [load_usage(p).astype(np.float64) for p in usage.split(',')]
    u = sum(x / np.maximum(1, x.sum(axis=1, keepdims=True)) for x in us)
    k = int(k)
    keep, cov = {}, []
    for il in range(N_PHYS):
        order = np.argsort(-u[il], kind='stable')[:k]
        keep[str(il)] = sorted(int(x) for x in order)
        cov.append(u[il][order].sum() / max(1, u[il].sum()))
    json.dump(keep, open(out, 'w'))
    print(f'K={k}: selections covered per layer min {min(cov):.3f} mean {np.mean(cov):.3f}')
    print(' '.join(f'{c:.2f}' for c in cov))


def cmd_mask(model, keep_path):
    from gguf import GGUFReader
    keep = json.load(open(keep_path))
    r = GGUFReader(model, 'r+')
    backup = {}
    for t in r.tensors:
        if not t.name.endswith('.exp_probs_b.bias'):
            continue
        il = t.name.split('.')[1]
        backup[t.name] = np.array(t.data)
        m = np.ones(t.data.shape[0], dtype=bool)
        m[keep[il]] = False
        t.data[m] = -1e9
    r.data.flush()
    np.savez(model + '.bias_backup.npz', **backup)
    print(f'masked {len(backup)} layers')


def cmd_unmask(model, backup_path):
    from gguf import GGUFReader
    b = np.load(backup_path)
    r = GGUFReader(model, 'r+')
    n = 0
    for t in r.tensors:
        if t.name in b:
            t.data[:] = b[t.name]
            n += 1
    r.data.flush()
    print(f'restored {n} layers')


def cmd_slice(model, keep_path, out):
    from gguf import GGUFReader, GGUFWriter, GGUFValueType
    keep = json.load(open(keep_path))
    k = len(keep['0'])
    r = GGUFReader(model)
    arch = r.fields['general.architecture'].contents()
    w = GGUFWriter(out, arch)
    for name, f in r.fields.items():
        if name.startswith('GGUF.') or name == 'general.architecture':
            continue
        val = f.contents()
        if name.endswith('.expert_count'):
            n_expert = int(val)
            val = k
        vt = f.types[0]
        sub = f.types[-1] if vt == GGUFValueType.ARRAY else None
        w.add_key_value(name, val, vt, sub_type=sub)
    sliced = 0
    for t in r.tensors:
        shape = [int(x) for x in t.shape]          # ggml order, expert dimension last
        raw = np.asarray(t.data).reshape(-1).view(np.uint8)
        parts = t.name.split('.')
        if parts[0] == 'blk' and shape and shape[-1] == n_expert and parts[1] in keep:
            per = raw.size // n_expert
            assert per * n_expert == raw.size, t.name
            raw = raw.reshape(n_expert, per)[keep[parts[1]]].reshape(-1)
            shape[-1] = k
            sliced += 1
        # int8 view: the writer then takes raw_shape as the element shape instead of deriving it from bytes
        w.add_tensor(t.name, raw.view(np.int8), raw_shape=list(reversed(shape)), raw_dtype=t.tensor_type)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f'sliced {sliced} tensors to {k} experts -> {out}')


if __name__ == '__main__':
    {'keep': cmd_keep, 'mask': cmd_mask, 'unmask': cmd_unmask, 'slice': cmd_slice}[sys.argv[1]](*sys.argv[2:])
