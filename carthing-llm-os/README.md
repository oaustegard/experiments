# carthing-llm-os

An attempt to write a small operating system from scratch, with no Linux underneath, to run a 4-bit LLM on the
Spotify Car Thing. The Car Thing is a discontinued dashboard controller with an Amlogic S905D2 (four Cortex-A53
cores) and 512 MB of RAM. The question is whether Qwen3-0.6B (0.6 billion parameters, stored at roughly 4 bits
each) can run on it, and how fast, once the OS is out of the way. The idea came from a Tom's Hardware story about
someone who deleted Linux and ran Qwen3 on a 24 KB x86-64 assembly OS. That project is x86 code, so none of it
can run on this ARM board.

**Status (2026-11-28): scaffold, runs under QEMU only.** QEMU is an emulator that imitates an ARM board on an
ordinary computer; nothing has run on a real Car Thing. A 23 KB kernel boots on QEMU's `virt` machine (Cortex-A53
×4, 512 MB). It turns on the MMU (memory management unit) and caches, starts all four cores through PSCI (an Arm
standard call for starting cores), and checks its 4-bit matrix-vector kernels against reference values computed in llama.cpp's formats. Every
check passes, both when it boots at EL1 and when it boots at EL2 ([`results/qemu-el1.log`](results/qemu-el1.log),
[`results/qemu-el2.log`](results/qemu-el2.log)). U-Boot is expected to hand over at EL2 on the Car Thing (both
levels are explained under Boot). No model runs yet, so there is no speed result.

## Kernel components and checks

1. **Boot.** The image starts with the header that arm64 Linux kernels use, so any loader that can start Linux can
   start it, including QEMU's `-kernel` and U-Boot's `booti` (U-Boot is a common bootloader, the program that loads
   a kernel). The loader hands over with the caches off, at EL2 or EL1. These are ARM privilege levels: kernels
   normally run at EL1, and EL2 is the level above, for hypervisors. The kernel drops from EL2 to EL1 if needed and
   turns on FP/SIMD (floating-point and vector instructions) (`src/boot.S`).
2. **MMU.** An identity map (each address maps to itself) with 4 KB pages: the 1 GB slot of address space where
   RAM sits is split into 2 MB blocks, so only real RAM is cacheable and everything else is device memory, which
   the CPU does not cache (`src/mmu.c`). Without the caches an A53 runs many times slower, so no kernel speed can
   be judged until the MMU is on.
3. **Cores.** Cores 1–3 are started with PSCI `CPU_ON`. The call goes through HVC or SMC (calls into firmware),
   whichever the boot EL implies. The Car Thing build always uses SMC, assuming (unchecked) that its firmware
   handles PSCI. Work is split across cores through a single job slot, with `wfe`/`sev` waits (sleep and wake-up
   instructions) (`src/smp.c`).
4. **Kernels.** llama.cpp's Q4_0 (4-bit) format for weights and Q8_0 (8-bit) for activations, the values passed
   between layers, with the block layout of GGUF files, llama.cpp's model file format; no real GGUF has been loaded
   yet. The code is the two quantizers, which turn floats into those blocks, and a scalar (plain C) and a NEON (ARM
   vector) `vec_dot_q4_0_q8_0` (`src/q4.c`). The A53 implements ARMv8.0, which has no `sdot` (8-bit dot-product)
   instruction, so the NEON version widens with `smull` and accumulates pairwise.
5. **Checks.** At boot, the kernel checks itself against test vectors in a binary file (a blob) that
   `tools/gen_testvec.py` writes with gguf-py, llama.cpp's own Python implementation of its quantizers. QEMU loads
   the blob at a fixed address, as U-Boot would load the weights on the device.

| check (from `results/qemu-el1.log`) | result |
|---|---|
| cores started through PSCI | 4 of 4 |
| NEON dot vs scalar dot, lengths 32–4096 | worst relative error 3.5e-7 |
| kernel's Q4_0 quantizer vs gguf-py, 32,000 blocks | byte-identical |
| kernel's Q8_0 quantizer vs gguf-py (round-half ties may differ by 1) | identical |
| 1000×1024 matrix-vector product vs the llama.cpp formula: scalar / NEON / NEON on 4 cores | worst relative error 1.4e-5 / 6.5e-5 / 6.5e-5 |

The log also prints timings and memory bandwidth. **Under QEMU those numbers describe the emulator and say nothing
about the Car Thing.** QEMU's software CPU emulation reports 17 GB/s of read bandwidth, which is far above anything
an A53's LPDDR4 delivers. They mean something only on the device.

The quantizers compile with `#pragma clang fp contract(off)`. With it on, clang fuses `x*id + 8.5` (`id` is the
inverse of the block scale) into one fused multiply-add, which rounds once instead of twice; one value in the test
set then truncates the other way, and 1 Q4_0 block in 32,000 differs from gguf-py.

## Run it

Needs clang, ld.lld, llvm-objcopy, `qemu-system-aarch64`, and Python with `numpy` and `gguf`.

```sh
make run         # boot at EL1, as QEMU does by default
make run-el2     # boot at EL2, which U-Boot is expected to use on hardware
make PLATFORM=carthing   # Car Thing build; untested
```

`make` writes the test-vector blob (`build/testvec.bin`) and the kernel (`build/<platform>/kernel.bin`). A QEMU
run takes about a minute. It ends with `RESULT: PASS` or `RESULT: FAIL (n)`, where n counts failed checks, and powers off through PSCI `SYSTEM_OFF`, which exits QEMU.

## Getting it onto a Car Thing (untested)

`platform/carthing.h` holds the addresses for the device. They come from the Amlogic G12A memory map and from
the community superbird U-Boot (superbird is the Car Thing's codename), and need checking first against a UART log
(the serial console output): the UART at `0xff803000`, RAM from `0x0`, the kernel at `0x01080000` and the blob at
`0x08000000`. The planned path is to load a custom U-Boot over USB in Amlogic's burn mode (a USB flashing mode)
with superbird-tool, then `mmc read` (or `ext4load`) the blob to `0x08000000` and `booti` the kernel. On hardware,
the kernel's output appears only on the UART pads (115200 8N1), so reading it means soldering wires to them. The
open risk is the boot chain: nobody has checked whether the Car Thing's secure boot, which verifies signatures on
the code it starts, lets unsigned code run past U-Boot.

## Next steps

1. Convert a real GGUF. Parse Qwen3-0.6B Q4_0 on the host into a flat blob (the tensors in order, plus a table of
   their offsets), and keep the tokenizer tables in it as well.
2. Write the forward pass (the model's computation for one token) on one core: RMSNorm, RoPE, QK-norm,
   grouped-query attention with a KV cache (the stored attention state of earlier tokens), SwiGLU, and the output
   layer, which shares its weights with the embedding table. Check its logits against llama.cpp on the same prompt.
3. Run it on all four cores, then on the device, and measure tokens per second against the read bandwidth.

Memory budget: the Q4_0 weights take about 340 MB (596M parameters at 4.5 bits each), which leaves about 170 MB of
the 512 MB for the KV cache, the activations, the kernel and whatever the firmware reserves.

## Prior art

Web searches did not find anyone running an LLM on a Car Thing, with or without Linux. The closest
projects found:

- **[NightRun](https://github.com/hardrave/NIGHTRUN)** boots a Raspberry Pi 5 or an x86-64 PC straight into a local LLM. It runs as a UEFI
  application (a program started by the computer's standard firmware), so it uses firmware services for input,
  storage and display and is not bare metal (running directly on the hardware with no OS); its authors call it "no
  conventional OS".
- **[VC4LLM](https://github.com/Foadsf/vc4llm)** is a bare-metal-style engine for the 32-bit Raspberry Pi 3B.
- **[arXiv 2601.03324](https://arxiv.org/abs/2601.03324)** (Bare-Metal Tensor Virtualization) studies bare-metal ARM64 inference with a custom memory arena.

This kernel differs from those in its target (a 512 MB device that U-Boot starts, with no UEFI) and in its plan to
have U-Boot put the weights in RAM before the kernel starts, so it needs no storage driver.

## Layout

| path | what |
|---|---|
| `src/boot.S` | Image header, EL2→EL1 drop, per-core stacks, exception vectors |
| `src/mmu.c` | page tables and MMU enable; built with `-mstrict-align` because it runs with the MMU off |
| `src/smp.c` | PSCI calls, secondary-core loop, `smp_run` |
| `src/q4.c`, `src/q4.h` | Q4_0 and Q8_0 formats, quantizers, scalar and NEON dot products, row-split matvec |
| `src/main.c` | boot sequence, self-tests, bandwidth probe |
| `src/uart.c` | PL011 (QEMU) and Meson (Car Thing) UART output, `kprintf`, exception report |
| `platform/*.h` | per-board addresses |
| `tools/gen_testvec.py` | writes the gguf-py test-vector blob |
| `Makefile`, `src/link.ld` | build, QEMU run targets, linker script (addresses come from `platform/*.h`) |
| `results/` | the QEMU logs the check table quotes |
