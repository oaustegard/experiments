"""Shared plumbing for the ALTA residual-code experiment.

Holds four things the trainer, the plotter and the tests all need:

* locating and importing the ALTA checkout (Shaw et al. 2024, arXiv 2410.18077,
  Apache 2.0) without hardcoding a path;
* a registry of the three compiled programs studied here, each with a fixed
  sequence length, an input sampler and a reference decode;
* the residual-stream instrumentation -- which dimensions any weight reads,
  which are ever non-zero on a trajectory, per-dimension decision tolerances,
  and the teacher trajectory itself;
* the compressed forward pass, which is the compiled forward pass with a code
  ``U`` (D x d) on every residual write and a readout ``R`` on every residual
  read.

At ``U = R = I`` the compressed pass is the compiled pass, exactly; the test
module asserts that.
"""
from __future__ import annotations

import os
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _lib.paths import spoke  # noqa: E402

# Threshold below which a residual coordinate counts as zero. Compiled ALTA
# attention is a softmax with a logit gap of `attention_scalar ** 2` = 1e4, so
# unselected positions contribute mass around e^-1e4; the residual is one-hot
# up to that, not exactly.
ZERO = 1e-12

# Activation budget for the compiled MLP's lookup layer; see `tolerances`.
TOLERANCE_BUDGET = 0.01


def alta_root() -> Path:
    """Return the ALTA checkout directory.

    ``ALTA_ROOT`` wins when set; otherwise the spoke root and the home
    directory are probed, in that order.
    """
    env = os.environ.get("ALTA_ROOT")
    candidates = [Path(env)] if env else []
    candidates += [
        spoke("alta"),
        Path(__file__).resolve().parents[2] / "alta",
        Path.home() / "alta",
    ]
    for candidate in candidates:
        if (candidate / "framework" / "compiler").is_dir():
            return candidate
    raise FileNotFoundError(
        "no ALTA checkout found; set ALTA_ROOT or clone "
        "google-deepmind/alta next to this repo"
    )


def import_alta():
    """Put ALTA on ``sys.path`` and return the modules used here."""
    root = str(alta_root())
    if root not in sys.path:
        sys.path.insert(0, root)
    from examples import parity, subleq  # noqa: PLC0415
    from framework import program  # noqa: PLC0415
    from framework import var_utils  # noqa: PLC0415
    from framework.compiler import compiler_config, compiler_utils, dim_utils  # noqa: PLC0415
    from framework.interpreter import interpreter_utils, program_utils  # noqa: PLC0415
    from framework.transformer import transformer_utils  # noqa: PLC0415

    return dict(
        parity=parity,
        subleq=subleq,
        program=program,
        var_utils=var_utils,
        compiler_config=compiler_config,
        compiler_utils=compiler_utils,
        dim_utils=dim_utils,
        interpreter_utils=interpreter_utils,
        program_utils=program_utils,
        transformer_utils=transformer_utils,
    )


# --------------------------------------------------------------------------
# Program registry
# --------------------------------------------------------------------------


@dataclass
class Case:
    """One compiled ALTA program plus everything needed to exercise it."""

    name: str
    kind: str  # "looped" or "feed-forward"
    build: Callable[[], Any]
    sample: Callable[[random.Random], list[int]]
    reference: Callable[[list[int], list[Any]], bool]
    num_layers: int
    training: Callable[[], list[list[int]]]
    config_kwargs: dict = field(default_factory=dict)
    note: str = ""


SUBLEQ_LAYERS = 14


def _subleq_case():
    alta = import_alta()
    subleq = alta["subleq"]

    def build(a, b, z, mem_a, mem_b) -> list[int]:
        """One instance of the Wikipedia SUBLEQ ADD routine.

        Instructions occupy positions 0-8; the three data cells `a`, `b` and
        `z` sit at a distinct triple of positions 9-15. Instances differ in
        both the addresses the machine chases and the values it subtracts.
        """
        mem = [0] * subleq.NUM_POSITIONS
        mem[0:3] = [a, z, 3]
        mem[3:6] = [z, b, 6]
        mem[6:9] = [z, z, -1]
        mem[a] = mem_a
        mem[b] = mem_b
        mem[z] = 0
        return subleq.encode_inputs(mem)

    # Data addresses are a rotation of positions 9-15, so every address is
    # seen in every role; addends run 0-4, so every reachable memory value is
    # seen by a training set of 63 instances.
    rotations = [(9 + i, 9 + (i + 1) % 7, 9 + (i + 2) % 7) for i in range(7)]
    addends = [(a, b) for a in range(5) for b in range(5)]
    diagonal = [(0, 0), (1, 3), (2, 2), (3, 1), (4, 4)]

    def sample(rng: random.Random) -> list[int]:
        a, b, z = rng.choice(rotations)
        ma, mb = rng.choice(addends)
        return build(a, b, z, ma, mb)

    def training() -> list[list[int]]:
        """A covering training set: every address in every role, every value.

        Random draws saturate short of what the evaluation set reaches, and a
        code cannot be asked to preserve a feature it never saw, so the
        training set is enumerated rather than sampled: every addend pair at
        one address rotation, covering every reachable memory value, and a
        diagonal of addend pairs at every rotation, covering every address in
        every role.
        """
        first = [build(*rotations[0], ma, mb) for ma, mb in addends]
        rest = [build(a, b, z, ma, mb) for a, b, z in rotations for ma, mb in diagonal]
        return first + rest

    def reference(input_ids, outputs):
        mem = subleq.decode_outputs([int(x) for x in input_ids])
        a, b = mem[0], mem[4]
        expected = mem[a] + mem[b]
        return subleq.decode(int(outputs[b])) == expected

    return Case(
        name="subleq",
        kind="looped",
        build=subleq.build_program_spec_sparse,
        sample=sample,
        reference=reference,
        num_layers=SUBLEQ_LAYERS,
        training=training,
        note="one-instruction computer, 16 positions, values in [-16, 16]",
    )


PARITY_LENGTH = 12


def _parity_seq_case():
    alta = import_alta()
    parity = alta["parity"]

    def sample(rng: random.Random) -> list[int]:
        return [rng.choice([0, 1]) for _ in range(PARITY_LENGTH)]

    def training() -> list[list[int]]:
        """Every prefix parity at every position, plus random strings."""
        rng = random.Random(1)
        fixed = [[0] * PARITY_LENGTH, [1] * PARITY_LENGTH]
        fixed += [[int(j <= i) for j in range(PARITY_LENGTH)] for i in range(PARITY_LENGTH)]
        return fixed + [sample(rng) for _ in range(12)]

    def reference(input_ids, outputs):
        return int(outputs[-1]) == sum(input_ids) % 2

    return Case(
        name="parity_seq",
        kind="looped",
        build=lambda: parity.build_sequential_program_absolute(
            max_input_length=PARITY_LENGTH, generate_rules=True
        ),
        sample=sample,
        reference=reference,
        num_layers=PARITY_LENGTH + 4,
        training=training,
        note="running parity, one loop iteration per input position",
    )


def _parity_ff_case():
    alta = import_alta()
    parity = alta["parity"]

    def sample(rng: random.Random) -> list[int]:
        bits = [rng.choice([0, 1]) for _ in range(PARITY_LENGTH)]
        return bits + [parity.EOS_VALUE]

    def training() -> list[list[int]]:
        """Every count of ones, so every bucket of the selector-width scalar."""
        rng = random.Random(1)
        fixed = [
            [1] * k + [0] * (PARITY_LENGTH - k) + [parity.EOS_VALUE]
            for k in range(PARITY_LENGTH + 1)
        ]
        return fixed + [sample(rng) for _ in range(12)]

    def reference(input_ids, outputs):
        bits = [x for x in input_ids if x != parity.EOS_VALUE]
        return int(outputs[-1]) == sum(bits) % 2

    return Case(
        name="parity_ff",
        kind="feed-forward",
        build=lambda: parity.build_sum_mod_2_program_spec(
            max_input_length=PARITY_LENGTH + 1, generate_rules=True
        ),
        sample=sample,
        reference=reference,
        num_layers=4,
        training=training,
        config_kwargs=dict(expansion_scalar_1=1000.0),
        note="selector-width count of ones, then one MLP; fixed depth",
    )


_BUILDERS = {
    "subleq": _subleq_case,
    "parity_seq": _parity_seq_case,
    "parity_ff": _parity_ff_case,
}

CASE_ORDER = ("subleq", "parity_seq", "parity_ff")


def get_case(name: str) -> Case:
    return _BUILDERS[name]()


# --------------------------------------------------------------------------
# Compiled model
# --------------------------------------------------------------------------


@dataclass
class Compiled:
    case: Case
    spec: Any
    params: Any
    mappings: Any
    dim: int


def compile_case(case: Case) -> Compiled:
    alta = import_alta()
    spec = case.build()
    config = alta["compiler_config"].Config(**case.config_kwargs)
    params = alta["compiler_utils"].compile_transformer(spec, config)
    mappings = alta["dim_utils"].get_var_mapping(spec)
    return Compiled(
        case=case,
        spec=spec,
        params=params,
        mappings=mappings,
        dim=mappings.end_idx,
    )


def var_slices(compiled: Compiled) -> dict[str, tuple[int, int]]:
    """Return ``{var_name: (start, end)}`` over residual dimensions."""
    alta = import_alta()
    dim_utils = alta["dim_utils"]
    out = {}
    for name, mapping in compiled.mappings.var_mappings.items():
        if isinstance(mapping, dim_utils.NumericalVarDimMapping):
            out[name] = (mapping.idx, mapping.idx + 1)
        else:
            out[name] = (mapping.start_idx, mapping.end_idx)
    return out


def read_dims(compiled: Compiled) -> np.ndarray:
    """Boolean mask of residual dimensions that some frozen weight reads.

    A dimension is read if it has a non-zero row in any query, key or value
    projection, in the first FFN weight matrix, or in the output transform.
    Everything else can be written but never influences the computation.
    """
    d = compiled.dim
    mask = np.zeros(d, dtype=bool)
    for head in compiled.params.attention_heads:
        for mat in (head.query_transform, head.key_transform, head.value_transform):
            mask |= np.abs(np.asarray(mat)).sum(axis=1) > 0
    mask |= np.abs(np.asarray(compiled.params.feed_forward_layers[0].weights)).sum(axis=1) > 0
    mask |= np.abs(np.asarray(compiled.params.output_transform)).sum(axis=1) > 0
    return mask


def tolerances(compiled: Compiled) -> np.ndarray:
    """Per-dimension absolute tolerance, from the machine's own thresholds.

    ALTA's compiled MLP is a lookup table: a rule with `K` antecedent atoms
    fires through `clipped_relu(sum of K indicators + 1 - K)`, so it reads 1
    when all `K` hold and at most 0 otherwise. An error of `e` on each
    indicator moves that activation by up to `K * e`, and the activation is
    multiplied straight into the residual write, so the per-coordinate
    tolerance has to be the activation budget divided by the widest rule.
    The budget is 0.01, so an indicator gets `0.01 / K_max`. A numerical
    coordinate is discretized into buckets whose boundaries are midpoints
    between declared values; a quarter of the tightest gap, over the same
    `K_max`. A tenth of that budget changes no result reported here, and a
    budget of 0.25 -- the naive half-of-half-a-unit -- admits codes that
    satisfy every constraint and still compute nothing, because dozens of
    rules one atom short of firing each contribute their own error.

    A looser tolerance is not merely imprecise, it is unsound: a code that
    satisfies a 0.25 indicator tolerance on every read coordinate can still
    drive the machine to the wrong answer, which is how the first version of
    this objective reported a satisfied loss on a code that computed nothing.
    """
    alta = import_alta()
    dim_utils = alta["dim_utils"]
    arity = max(len(rule.lhs) for rule in compiled.spec.mlp.get_rules())
    tol = np.full(compiled.dim, TOLERANCE_BUDGET / arity)
    for name, mapping in compiled.mappings.var_mappings.items():
        if not isinstance(mapping, dim_utils.NumericalVarDimMapping):
            continue
        values = compiled.spec.variables[name].values
        if not values or len(values) < 2:
            continue
        gaps = np.diff(np.asarray(sorted(values), dtype=float))
        tol[mapping.idx] = 0.25 * float(gaps.min()) / arity
    return tol


# --------------------------------------------------------------------------
# Forward passes
# --------------------------------------------------------------------------


def _tu():
    return import_alta()["transformer_utils"]


_FFN_CACHE: dict[int, list] = {}


def _sparse_ffn(params):
    """CSR copies of the MLP's weight matrices, keyed by identity.

    ALTA's compiled MLP is a lookup table: SUBLEQ's has 43,127 rules, so its
    third weight matrix is 402 x 43,127, and a dense forward pass through it
    costs a gigaflop per layer per input. Every rule reads at most `K_max`
    coordinates and writes at most two, so those matrices are about 99.97%
    zero, and the same arithmetic in CSR is two orders of magnitude cheaper.
    The result is identical, not approximate; `test_sparse_ffn_matches_alta`
    asserts that against ALTA's own `run_ffn`.
    """
    cached = _FFN_CACHE.get(id(params))
    if cached is None:
        layers = params.feed_forward_layers
        cached = [
            (
                sparse.csr_matrix(np.asarray(layer.weights).T),
                np.asarray(layer.biases),
                idx == len(layers) - 1,
            )
            for idx, layer in enumerate(layers)
        ]
        _FFN_CACHE[id(params)] = cached
    return cached


def run_ffn(compiled: Compiled, activations: np.ndarray) -> np.ndarray:
    """The compiled MLP sub-layer on a (batch * positions, dim) block."""
    out = np.ascontiguousarray(activations.T)
    for weights, biases, is_last in _sparse_ffn(compiled.params):
        out = weights @ out + biases[:, None]
        if not is_last:
            out = np.minimum(1.0, np.maximum(0.0, out))
    return np.ascontiguousarray(out.T)


def _relative_mask(compiled: Compiled, head, n_pos: int) -> np.ndarray:
    return _tu().get_relative_position_embeddings(head.relative_position_mask, n_pos)


def batched_attention(compiled: Compiled, states: np.ndarray) -> np.ndarray:
    """Multi-head attention on a (batch, positions, dim) block.

    Same arithmetic as ALTA's `multihead_attention`, one sequence at a time
    replaced by one batched matmul, because the sweep runs the model on a
    hundred inputs at every width.
    """
    out = np.zeros_like(states)
    n_pos = states.shape[1]
    for head in compiled.params.attention_heads:
        queries = states @ np.asarray(head.query_transform)
        keys = states @ np.asarray(head.key_transform)
        logits = queries @ keys.transpose(0, 2, 1)
        logits = logits + _relative_mask(compiled, head, n_pos)
        logits = logits - logits.max(axis=-1, keepdims=True)
        weights = np.exp(logits)
        weights /= weights.sum(axis=-1, keepdims=True)
        values = states @ np.asarray(head.value_transform)
        out += (weights @ values) @ np.asarray(head.output_transform)
    return out


def forward(compiled: Compiled, batch: list[list[int]], code=None, readout=None,
            collect=False):
    """Run the compressed model on a batch of inputs.

    ``z = x @ U`` is the compressed state; every weight that reads the residual
    sees ``z @ R.T`` instead of ``x``, and every weight that writes it has its
    output multiplied by ``U``. Residual connections are additions in the
    compressed space, so at ``U = R = I`` this is the compiled forward pass
    with no change at all.

    Returns decoded outputs, and with ``collect`` also the trajectory: ``pre``
    is the state each layer sees at its attention sub-layer and ``mid`` the
    state after the attention residual connection, which is what the MLP
    reads.
    """
    tu = _tu()
    params = compiled.params
    identity = np.eye(compiled.dim)
    code = identity if code is None else np.asarray(code)
    readout = identity if readout is None else np.asarray(readout)
    states = np.stack([tu.initialize_embeddings(params, ids) for ids in batch])
    z = states @ code
    n_batch, n_pos = z.shape[0], z.shape[1]
    pre, mid = [], []
    for _ in range(compiled.case.num_layers):
        x = z @ readout.T
        if collect:
            pre.append(x.copy())
        z = z + batched_attention(compiled, x) @ code
        x = z @ readout.T
        if collect:
            mid.append(x.copy())
        flat = run_ffn(compiled, x.reshape(n_batch * n_pos, -1))
        z = z + flat.reshape(n_batch, n_pos, -1) @ code
    logits = (z @ readout.T) @ np.asarray(params.output_transform)
    outputs = np.argmax(logits, axis=-1).tolist()
    if collect:
        return outputs, dict(pre=np.array(pre), mid=np.array(mid))
    return outputs


def trace(compiled: Compiled, input_ids: list[int]) -> dict[str, np.ndarray]:
    """Residual stream the compiled model visits on one input."""
    _, states = forward(compiled, [input_ids], collect=True)
    return dict(pre=states["pre"][:, 0], mid=states["mid"][:, 0])


def trace_batch(compiled: Compiled, batch: list[list[int]]) -> dict[str, np.ndarray]:
    """Same, for a batch: arrays are (layers, batch, positions, dim)."""
    _, states = forward(compiled, batch, collect=True)
    return states


def run_compiled(compiled: Compiled, input_ids: list[int]) -> list[int]:
    return forward(compiled, [input_ids])[0]


def run_compiled_reference(compiled: Compiled, input_ids: list[int]) -> list[int]:
    """ALTA's own forward pass, used to check the batched sparse one."""
    tu = _tu()
    return tu.run_transformer(
        compiled.params,
        learned_ffn_params=None,
        input_ids=input_ids,
        max_layers=compiled.case.num_layers,
    )


def run_compressed(compiled: Compiled, input_ids: list[int], code, readout) -> list[int]:
    return forward(compiled, [input_ids], code, readout)[0]


def run_interpreter(compiled: Compiled, input_ids: list[int]) -> list[Any]:
    alta = import_alta()
    activations = alta["program_utils"].initialize_activations(compiled.spec, input_ids)
    return alta["interpreter_utils"].run_transformer(
        compiled.spec, activations, max_layers=compiled.case.num_layers
    )


def sample_inputs(case: Case, n: int, seed: int, exclude=()) -> list[list[int]]:
    """Sample `n` distinct inputs, avoiding anything in `exclude`."""
    rng = random.Random(seed)
    seen, out = {tuple(x) for x in exclude}, []
    while len(out) < n:
        ids = case.sample(rng)
        key = tuple(ids)
        if key in seen:
            continue
        seen.add(key)
        out.append(ids)
    return out


def training_inputs(case: Case) -> list[list[int]]:
    """Return the case's training inputs, deduplicated in order."""
    seen, out = set(), []
    for ids in case.training():
        key = tuple(ids)
        if key not in seen:
            seen.add(key)
            out.append(ids)
    return out
