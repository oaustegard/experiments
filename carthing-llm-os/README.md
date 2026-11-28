# carthing-llm-os

A small operating system written from scratch, with no Linux underneath, to run a 4-bit LLM on the Spotify Car
Thing. The Car Thing is a discontinued dashboard controller with an Amlogic S905D2 (four Cortex-A53 cores) and
512 MB of RAM. The question is whether Qwen3-0.6B (0.6 billion parameters, stored at about 4.5 bits each) can run
on it, and how fast, once the OS is out of the way. The idea came from a Tom's Hardware story about someone who
deleted Linux and ran Qwen3 on a 24 KB x86-64 assembly OS. That project is x86 code, so none of it can run on this
ARM board. No Linux baseline (llama.cpp on the stock firmware) has been measured either.

**Status (2026-11-28): Qwen3-0.6B runs in the kernel under QEMU; nothing has run on a real Car Thing.** QEMU is an
emulator that imitates an ARM board on an ordinary computer; its `virt` machine matches the Car Thing's core type,
core count and RAM size (four Cortex-A53 cores, 512 MB), not its memory map or peripherals. A 37 KB kernel image
(plus 62 MB of buffers it zeroes at boot) finds a 337 MB model blob already placed in RAM, and
completes the prompt "The capital of France is Paris. The capital of Germany is" with " Berlin. The capital of
Italy is Rome. The capital of Japan is Tokyo." at EL1 and at EL2 ([`results/qemu-model-el1.log`](results/qemu-model-el1.log),
[`results/qemu-model-el2.log`](results/qemu-model-el2.log)). Its logits match a native build of the same C code
within that arithmetic's measured rounding noise. QEMU runs it at 0.6 s per token,
which measures the emulator; the speed on the device is unknown until it runs there.

## How the model runs

Q4_0 and Q8_0 are llama.cpp's block formats: 32 values share one f16 scale, stored as 4-bit or 8-bit integers, so
Q4_0 costs 4.5 bits per weight. `tools/convert.py` turns the Hugging Face weights (`Qwen/Qwen3-0.6B`) into one flat
blob: every matrix, the embedding table included,
quantized to llama.cpp's Q4_0 format by gguf-py (llama.cpp's own Python quantizers), norm weights as f32, a RoPE
cos/sin table for 512 positions, and the vocabulary as raw bytes for printing. Quantizing every matrix to Q4_0 is
the choice llama.cpp's `llama-quantize --pure` makes; the output was not compared with that tool's byte for byte. The blob also carries the prompt's token ids and
reference outputs for the kernel's self-check.

`src/qwen3.c` is the forward pass for one token: embedding row, then 28 layers of RMSNorm, the Q/K/V projections,
per-head QK-norm, RoPE, attention over a KV cache (16 query heads sharing 8 key/value heads), the output
projection, and a SwiGLU feed-forward block, then the final norm and the output layer, which reuses the embedding
table. Before each matrix product the input is quantized to Q8_0, as llama.cpp does for Q4_0 weights, and the KV
cache is stored in f16. It is plain C with no libc (including its own `expf`), so the kernel and a native host
build compile the same file. In the kernel every matrix product is split by rows across the four cores and uses
the NEON (ARM SIMD) dot product.

Memory: the kernel's buffers are static, and the KV cache for a 512-token context is 58.7 MB of the kernel's 62 MB
of zero-initialised data. In QEMU the blob sits at `0x48200000`, past the device tree QEMU places at `0x48000000`.

## Correctness

| comparison | logits | top-1 token agreement |
|---|---|---|
| C, exact build (float activations, f32 KV cache) vs PyTorch on the same Q4_0 weights, 115-token paragraph | max difference 1.4e-4 | 100% |
| C, device arithmetic (Q8_0 activations, f16 KV cache) vs PyTorch, same paragraph | max difference 1.35, mean KL 0.0034 | 95.7% |
| kernel in QEMU (NEON, 4 cores) vs the host build, 28 teacher-forced positions | max difference 0.30 over the host build's top 8 tokens | 27 of 28 |
| two host builds that differ only in FMA contraction (fused multiply-add), same 28 positions | max difference 0.57 over all tokens | 100% |
| Q4_0 model vs the original bf16 weights, same paragraph | — | 71% |

The host build is `tools/host_run.c`: the same C code and rounding as the kernel, with scalar dot products, built
with clang's default FMA setting on an x86 host. The 28 positions are the 12 prompt tokens plus PyTorch's 16-token
continuation, fed in whatever the model would have picked. The paragraph is `EVAL_TEXT` in `tools/convert.py`. Rows
1, 2 and 4 come from [`results/host_check.json`](results/host_check.json), row 3 from the model logs, and row 5
(PyTorch with float activations on both sides) from [`results/convert.json`](results/convert.json). The exact build shows that the C code computes what PyTorch computes. The gap in the second row
is the cost of Q8_0 activations and the f16 cache. The fourth row is the noise floor of that arithmetic: each Q8_0
re-quantization can turn a 1-ulp difference into a rounding flip, so the one pair of builds measured,
differing only in the fusing of float operations, disagrees by up to 0.57 logits. The kernel's 0.30 falls inside it,
and its check passes when the worst difference is under 1.0 with at most two top-1 disagreements. A real bug, such
as wrong RoPE or a wrong head mapping, would be expected to move logits by whole units; no bug was injected to
confirm it.

Free-running generation can split where two tokens are nearly tied. On this prompt the last sentence differs: the host build writes
" The capital of Spain is Madrid." where PyTorch and the kernel write " The capital of Japan is Tokyo."

Quantizing every matrix to Q4_0 moves this small model's predictions. On the paragraph, the Q4_0 model's top token
agrees with the bf16 model's at 71% of positions, though on the prompt both give the same 16 tokens. llama.cpp's
usual Q4_0 files keep the embedding and output table at higher precision; that was not tried here.

## Scaffold checks

The kernel also checks its building blocks against a test-vector blob from `tools/gen_testvec.py`
([`results/qemu-el1.log`](results/qemu-el1.log), [`results/qemu-el2.log`](results/qemu-el2.log)):

| check | result |
|---|---|
| cores started through PSCI | 4 of 4 |
| NEON dot vs scalar dot, lengths 32–4096 | worst relative error 3.5e-7 |
| kernel's Q4_0 quantizer vs gguf-py, 32,000 blocks | byte-identical |
| kernel's Q8_0 quantizer vs gguf-py (round-half ties may differ by 1) | identical |
| 1000×1024 matrix-vector product vs the llama.cpp formula: scalar / NEON / NEON on 4 cores | worst relative error 1.4e-5 / 6.5e-5 / 6.5e-5 |

The quantizers compile with `#pragma clang fp contract(off)`. With it on, clang fuses `x*id + 8.5` (`id` is the
inverse of the block scale) into one fused multiply-add, which rounds once instead of twice, and 1 Q4_0 block in
32,000 then differs from gguf-py.

The logs also print timings and memory bandwidth. Under QEMU those numbers describe the emulator and say nothing
about the Car Thing: QEMU's software CPU emulation reports up to 17 GB/s of read bandwidth.

## Kernel components

1. **Boot.** The image starts with the header that arm64 Linux kernels use, so any loader that can start Linux can
   start it, including QEMU's `-kernel` and U-Boot's `booti` (U-Boot is the bootloader on boards like the Car
   Thing). The loader hands over with the caches off, at EL2 or EL1. These are ARM privilege levels: kernels
   normally run at EL1, and EL2 is the level above, for hypervisors, where U-Boot is expected to leave the Car
   Thing. The kernel drops from EL2 to EL1 if needed and turns on the floating-point and vector unit
   (`src/boot.S`). Every check passes from both levels.
2. **MMU.** An identity map (each address maps to itself) with 4 KB pages. The 1 GB of address space that holds
   RAM is split into 2 MB blocks, so only real RAM is cacheable and everything else is uncached device memory
   (`src/mmu.c`). On ARM the data cache works only with the MMU on, and an A53 without caches is too slow to judge.
3. **Cores.** Cores 1–3 are started with PSCI `CPU_ON`, the Arm standard firmware call for starting cores, made
   through HVC or SMC depending on the boot level. The Car Thing build always uses SMC, assuming (unchecked) that
   its firmware handles PSCI. Work is split across cores through a single job slot, with `wfe`/`sev` sleep and
   wake-up instructions (`src/smp.c`).
4. **Compute kernels.** Q4_0 (4-bit) weight blocks and Q8_0 (8-bit) activation blocks in llama.cpp's layout, the two
   quantizers, and a scalar and a NEON `vec_dot_q4_0_q8_0` (`src/q4.c`). The A53 implements ARMv8.0, which has no
   `sdot` (8-bit dot-product) instruction, so the NEON version widens with `smull` and accumulates pairwise.
5. **Model.** The Qwen3 forward pass (`src/qwen3.c`) and the model mode in `src/main.c`, which runs the prompt,
   generates 16 tokens, prints them, and then runs the logit check.

## Run it

Needs clang, ld.lld, llvm-objcopy and `qemu-system-aarch64`, plus Python with `numpy`, `gguf`, `torch`,
`transformers`, `safetensors` and `huggingface_hub`. Tested with clang 18.1.3, QEMU 8.2.2, gguf 0.19.0, torch 2.14.1
and transformers 5.24.1 on a 4-core x86 host; the QEMU flags are in the `Makefile`.

```sh
make check-host  # download Qwen3-0.6B, write the blob, check the C forward pass natively against PyTorch
make run-model   # boot the kernel in QEMU with the model (under a minute)
make run         # boot with the test-vector blob instead; make run-el2 boots at EL2
make PLATFORM=carthing   # Car Thing build; untested
```

`make check-host` also writes the host build's outputs into the blob, which the kernel's logit check needs. A run
ends with `RESULT: PASS` or `RESULT: FAIL (n)`, where n counts failed checks, and powers off through PSCI, which
exits QEMU. Downloads, the blob and the PyTorch reference go in `work/` (about 1.9 GB, gitignored).

## Getting it onto a Car Thing (untested)

`platform/carthing.h` holds the addresses for the device. They come from the Amlogic G12A memory map and from
the community superbird U-Boot (superbird is the Car Thing's codename), and need checking against a UART log (the
serial console output) first. They put the UART at `0xff803000`, RAM from `0x0`, the kernel at `0x01080000` and the
blob at `0x08000000`. The kernel and its buffers then occupy `0x01080000`–`0x04BBF000` and the model blob
`0x08000000`–`0x1C1BA3C0`. Whether the firmware reserves memory inside either range is not known; if it falls in the
kernel's range, a smaller `QW_MAX_CTX` in `src/main.c` (with `MAX_CTX` in `tools/convert.py`) shrinks the KV cache.

The planned path is to load a custom U-Boot over USB in Amlogic's burn mode with superbird-tool, read the model
blob from eMMC to `0x08000000`, and `booti` the kernel. The kernel's output appears only on the UART pads (115200
8N1), so reading it means soldering wires to them. The open risk is the boot chain: this project has not checked
whether the Car Thing's secure boot accepts the custom U-Boot and the unsigned kernel it starts.

## Next steps

1. Run it on a Car Thing and measure tokens per second against the read bandwidth the kernel reports.
2. Batch the prompt: prefill now runs one token at a time, at decode speed.
3. Keep the embedding and output table at Q8_0 and measure what it recovers of the 71% agreement with bf16, and
   what it costs in memory (165 MB instead of 88 MB for that table; the blob would then need to start lower than
   `0x08000000` to fit in 512 MB).

## Prior art

Web searches did not find anyone running an LLM on a Car Thing, with or without Linux. The closest projects found:

- **[NightRun](https://github.com/hardrave/NIGHTRUN)** boots a Raspberry Pi 5 or an x86-64 PC straight into a
  local LLM. It runs as a UEFI application (a program started by the computer's standard firmware), so it uses
  firmware services for input, storage and display; its authors call it "no conventional OS".
- **[VC4LLM](https://github.com/Foadsf/vc4llm)** is a bare-metal-style engine for the 32-bit Raspberry Pi 3B.
- **[arXiv 2601.03324](https://arxiv.org/abs/2601.03324)** (Bare-Metal Tensor Virtualization) studies bare-metal
  ARM64 inference with a custom memory arena.

This kernel differs from those in its target (a 512 MB device that U-Boot starts, with no UEFI) and in its plan to
have U-Boot put the weights in RAM before the kernel starts, so it needs no storage driver.

## Layout

| path | what |
|---|---|
| `src/boot.S` | Image header, EL2→EL1 drop, per-core stacks, exception vectors |
| `src/mmu.c` | page tables and MMU enable; built with `-mstrict-align` because it runs with the MMU off |
| `src/smp.c` | PSCI calls, secondary-core loop, `smp_run` |
| `src/q4.c`, `src/q4.h` | Q4_0 and Q8_0 formats, quantizers, scalar and NEON dot products, row-split matvec |
| `src/qwen3.c`, `src/qwen3.h` | blob format and the Qwen3 forward pass (portable C) |
| `src/main.c` | boot sequence, self-tests, model mode, bandwidth probe |
| `src/uart.c` | PL011 (QEMU) and Meson (Car Thing) UART output, `kprintf`, exception report |
| `platform/*.h` | per-board addresses |
| `tools/convert.py` | Hugging Face weights → model blob, plus the PyTorch reference |
| `tools/host_run.c`, `tools/check_host.py` | native build of the forward pass and its comparison with PyTorch |
| `tools/gen_testvec.py` | writes the gguf-py test-vector blob |
| `Makefile`, `src/link.ld` | build, QEMU run targets, linker script (addresses come from `platform/*.h`) |
| `results/` | QEMU logs and the JSON the tables quote |
