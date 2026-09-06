"""Fast guards for the latent-calculator experiment (pytest, monad only)."""

import json
import os
import sys

import pytest
import torch

import data as D
import eval as E
import model_utils as mu
import train_port as TP

HERE = os.path.dirname(os.path.abspath(__file__))
K = 8


@pytest.fixture(scope="module")
def lm():
    model, tok = mu.load_model("monad")
    return model, tok


@pytest.fixture(scope="module")
def rows():
    return D.build(seed=1, n_train=200, n_val=40, n_test=40)


# ---------------------------------------------------------------- data
def test_split_lengths(rows):
    for r in rows["train"] + rows["val"] + rows["test_in"]:
        assert all(le in D.TRAIN_LENGTHS for le in r["lengths"]), r
    for r in rows["test_len5"]:
        assert D.HELDOUT_LENGTH in r["lengths"], r


def test_answer_format(rows):
    for r in rows["train"]:
        assert r["answer"] == " " + r["result_string"] + "\n"
        assert r["result_string"] == D.compute(r["op"], r["a"], r["b"])[0]
        if r["op"] == "cmp":
            assert r["result_string"] in ("greater", "less", "equal")
        else:
            assert r["result_string"].lstrip("-").isdigit()


def test_cmp_equal_rate():
    s = D.build(seed=7, n_train=4000, n_val=10, n_test=10)
    cmps = [r for r in s["train"] if r["op"] == "cmp"]
    eq = sum(1 for r in cmps if r["a"] == r["b"]) / len(cmps)
    assert 0.02 < eq < 0.10, eq
    ops = {o: sum(1 for r in s["train"] if r["op"] == o) for o in D.OPS}
    assert min(ops.values()) > 0.2 * len(s["train"])


def test_token_budget(rows):
    allr = [r for v in rows.values() for r in v]
    worst = D.check_token_lengths(allr)
    assert all(v < D.MAX_TOKENS for v in worst.values()), worst


def test_deterministic():
    a = D.build(seed=3, n_train=50, n_val=10, n_test=10)
    b = D.build(seed=3, n_train=50, n_val=10, n_test=10)
    assert a == b


# ---------------------------------------------------------------- calculator
def test_calculator_basic():
    a = D.digits_right_aligned("123", 6)
    b = D.digits_right_aligned("45", 6)
    assert mu.decode_operand(a) == 123
    assert mu.calculate(D.OPS.index("add"), a, b) == "168"
    assert mu.calculate(D.OPS.index("sub"), b, a) == "-78"
    assert mu.calculate(D.OPS.index("mul"), a, b) == "5535"
    assert mu.calculate(D.OPS.index("cmp"), a, b) == "greater"
    assert mu.calculate(D.OPS.index("cmp"), a, a) == "equal"
    assert mu.calculate(D.OPS.index("cmp"), b, a) == "less"


def test_calculator_blanks():
    # BLANK inside a number counts as 0; all-blank operand is 0
    slots = [1, D.BLANK, 3, D.BLANK, D.BLANK, D.BLANK]   # 3_1 -> 301
    assert mu.decode_operand(slots) == 301
    assert mu.decode_operand([D.BLANK] * 6) == 0
    assert mu.calculate(D.OPS.index("add"), [D.BLANK] * 6, [5] + [D.BLANK] * 5) == "5"


def test_calculator_matches_reference(rows):
    for r in rows["train"][:100]:
        q = D.query_target(r["op"], r["a"], r["b"])
        got = mu.calculate(q["op"], q["a_slots"], q["b_slots"])
        assert got == r["result_string"], r


# ---------------------------------------------------------------- modules
def test_module_shapes_and_params():
    for hidden in (256, 576):
        qh = mu.QueryHead(hidden)
        enc = mu.ResultEncoder(hidden)
        n = mu.count_params(qh, enc)
        assert n < 2_000_000, (hidden, n)
        h = torch.randn(4, hidden)
        op, slot = qh(h)
        assert op.shape == (4, len(D.OPS))
        assert slot.shape == (4, 2 * D.N_OPERAND_SLOTS, D.N_DIGIT_CLASSES)
        v = enc(mu.result_symbols(["12", "-3", "greater", "0"]))
        assert v.shape == (4, hidden)


def test_result_symbols_roundtrip():
    s = mu.result_symbols(["-45", "greater"])
    assert s.shape == (2, mu.N_RESULT_TOKENS)
    assert s[0, 0].item() == 5 and s[0, 1].item() == 4
    assert s[0, D.N_RESULT_SLOTS].item() == mu.SIGN_OFFSET + 1
    assert s[1, D.N_RESULT_SLOTS + 1].item() == mu.KIND_OFFSET + D.KINDS.index("greater")


# ---------------------------------------------------------------- mechanics
def _batch(lm, rows, n=4):
    model, tok = lm
    return mu.encode_rows(tok, rows["val"][:n])


def test_forward_upper_matches_full(lm, rows):
    model, tok = lm
    b = _batch(lm, rows)
    with torch.no_grad():
        full = model(input_ids=b["input_ids"], attention_mask=b["attention_mask"],
                     output_hidden_states=True, use_cache=False)
        lg = mu.forward_upper(model, full.hidden_states[K + 1],
                              b["attention_mask"], K + 1)
    m = b["attention_mask"].bool()
    assert torch.allclose(lg[m], full.logits[m], atol=1e-4)


def test_residual_hook_touches_only_t(lm, rows):
    """The hook must add the vector at position t of layer k's output and
    nowhere else.  NOTE: output_hidden_states captures the PRE-hook value in
    transformers 5.16, so the layer output is captured with our own hook
    registered after the injection hook."""
    model, tok = lm
    b = _batch(lm, rows)
    hook = mu.attach_hook(model, K)
    grabbed = []
    cap = model.model.layers[K].register_forward_hook(
        lambda m, a, o: grabbed.append((o[0] if isinstance(o, tuple) else o).clone()))
    try:
        with torch.no_grad():
            model(input_ids=b["input_ids"], attention_mask=b["attention_mask"],
                  use_cache=False)
            with mu.injection("residual", K, torch.zeros(4, 256), b["t"]):
                model(input_ids=b["input_ids"],
                      attention_mask=b["attention_mask"], use_cache=False)
            vec = torch.randn(4, 256)
            with mu.injection("residual", K, vec, b["t"]):
                model(input_ids=b["input_ids"],
                      attention_mask=b["attention_mask"], use_cache=False)
    finally:
        cap.remove()
        hook.remove()
    base, zero, nz = grabbed
    assert torch.allclose(base, zero, atol=1e-6)
    d = nz - base
    for i in range(4):
        t = int(b["t"][i])
        assert torch.allclose(d[i, t], vec[i], atol=1e-5)
        other = torch.cat([d[i, :t], d[i, t + 1:]], 0)
        assert other.abs().max() < 1e-6


def test_kv_slot_visibility(lm, rows):
    model, tok = lm
    b = _batch(lm, rows)
    torch.manual_seed(0)
    vec = torch.randn(4, 256)
    ek, ev = mu.make_slot_kv(model, K, vec)
    with torch.no_grad():
        base = model(input_ids=b["input_ids"], attention_mask=b["attention_mask"],
                     use_cache=False).logits
        with mu.injection("kv", K, vec, b["t"], extra_k=ek, extra_v=ev):
            got = model(input_ids=b["input_ids"],
                        attention_mask=b["attention_mask"], use_cache=False).logits
    d = (got - base).abs().max(-1).values
    for i in range(4):
        t = int(b["t"][i])
        # tolerance, not 0: concatenating a key changes the softmax
        # reduction order, which is worth ~1e-5 after 64 layers
        assert d[i, :t].max() < 1e-3, d[i, :t].max()
        assert d[i, t] > 20 * d[i, :t].max().clamp(min=1e-6)


def test_delayed_is_residual_shifted(lm, rows):
    model, tok = lm
    b = _batch(lm, rows)
    hook = mu.attach_hook(model, K)
    vec = torch.randn(4, 256)
    try:
        with torch.no_grad():
            with mu.injection("residual", K, vec, b["t"] + 1):
                a = model(input_ids=b["input_ids"],
                          attention_mask=b["attention_mask"],
                          use_cache=False).logits
            with mu.injection("delayed", K, vec, b["t"]):
                c = model(input_ids=b["input_ids"],
                          attention_mask=b["attention_mask"],
                          use_cache=False).logits
    finally:
        hook.remove()
    assert torch.allclose(a, c, atol=1e-6)


@pytest.mark.parametrize("arm", ["none", "residual", "kv"])
def test_generation_matches_teacher_forcing(lm, rows, arm):
    model, tok = lm
    row = rows["val"][0]
    vec = torch.randn(1, 256) * 0.3 if arm != "none" else None
    hook = mu.attach_hook(model, K) if arm == "residual" else None
    try:
        _, _, gen_ids = E.generate(model, tok, [row["prompt"]], arm, K, vec,
                                   max_new=4, bs=1)
        b = mu.encode_rows(tok, [row], with_answer=False)
        t = int(b["t"][0])
        ek = ev = None
        if arm == "kv":
            ek, ev = mu.make_slot_kv(model, K, vec)
        with torch.no_grad():
            if arm == "none":
                lg = model(input_ids=b["input_ids"],
                           attention_mask=b["attention_mask"],
                           use_cache=False).logits
            else:
                with mu.injection(arm, K, vec, b["t"], extra_k=ek, extra_v=ev):
                    lg = model(input_ids=b["input_ids"],
                               attention_mask=b["attention_mask"],
                               use_cache=False).logits
        tf_first = int(lg[0, t].argmax())
    finally:
        if hook is not None:
            hook.remove()
    assert tf_first == gen_ids[0][0], (tf_first, gen_ids[0][0])


def test_delayed_generation_second_token(lm, rows):
    """Injection at t+1 must not change the first token but must be applied
    during the first decode step."""
    model, tok = lm
    row = rows["val"][0]
    torch.manual_seed(11)
    vec = torch.randn(1, 256) * 10.0   # large enough to move the argmax
    hook = mu.attach_hook(model, K)
    try:
        _, _, base_ids = E.generate(model, tok, [row["prompt"]], "none", K,
                                    None, max_new=4, bs=1)
        _, _, got_ids = E.generate(model, tok, [row["prompt"]], "delayed", K,
                                   vec, max_new=4, bs=1, stop=False)
        _, _, res_ids = E.generate(model, tok, [row["prompt"]], "residual", K,
                                   vec, max_new=4, bs=1, stop=False)
        base_ids, got_ids, res_ids = base_ids[0], got_ids[0], res_ids[0]
    finally:
        hook.remove()
    assert got_ids[0] == base_ids[0]      # t+1 injection cannot move token 0
    assert got_ids[1:] != base_ids[1:]    # but it must move the rest
    assert res_ids != base_ids            # the t injection moves the output


def test_text_arm_inserts_result():
    p = E.text_prompt("12 + 3 =", "15")
    assert p == "12 + 3 = [15]"


def test_gradients_reach_encoder_only(lm, rows):
    model, tok = lm
    enc = mu.ResultEncoder(256)
    sub = rows["train"][:4]
    b = mu.encode_rows(tok, sub)
    with torch.no_grad():
        h = model(input_ids=b["input_ids"], attention_mask=b["attention_mask"],
                  output_hidden_states=True, use_cache=False).hidden_states[K + 1]
    syms = mu.result_symbols([r["result_string"] for r in sub])
    logits = TP.run_batch(model, enc, b, h, K, "residual", syms)
    loss = mu.answer_token_loss(logits, b["labels"])
    loss.backward()
    assert all(p.grad is not None and p.grad.abs().sum() > 0
               for p in enc.parameters())
    assert all(p.grad is None for p in model.parameters())
    assert all(not p.requires_grad for p in model.parameters())


def test_resume_skips_completed_epoch(tmp_path, monkeypatch):
    jp = tmp_path / "journal.jsonl"
    with open(jp, "w") as f:
        f.write(json.dumps({"model": "monad", "arm": "residual", "epoch": 1}) + "\n")
    monkeypatch.setattr(TP, "journal_path", lambda: str(jp))
    done = TP.read_journal("monad", "residual")
    assert 1 in done
    enc = mu.ResultEncoder(256)
    torch.save({"state_dict": enc.state_dict()},
               TP.ckpt_path("monad", "residual", 1))
    monkeypatch.setattr(sys, "argv", ["train_port.py", "--model", "monad",
                                      "--arm", "residual", "--k", str(K),
                                      "--epochs", "1", "--resume"])
    TP.main()   # must return without training
