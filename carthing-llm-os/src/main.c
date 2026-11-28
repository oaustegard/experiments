#include "kernel.h"
#include "q4.h"
#include "qwen3.h"
#include <arm_neon.h>

/* Test-vector blob written by tools/gen_testvec.py and placed at BLOB_ADDR
 * (QEMU: -device loader; Car Thing: U-Boot loads it before booti). */
typedef struct {
    char magic[8];                    /* "Q4TV0001" */
    uint32_t rows, cols;
    uint32_t off_wf32, off_wq, off_x, off_xq, off_yref, off_yf32, total;
} testvec;

#define MAX_ROWS 4096
#define MAX_WQ   (MAX_ROWS * 1024 / QK)
static float y_out[MAX_ROWS];
static block_q8_0 xq_dev[4096 / QK];
static block_q4_0 wq_dev[MAX_WQ];
static int failures;

static void check(int ok, const char *what) {
    kprintf("  [%s] %s\n", ok ? "pass" : "FAIL", what);
    if (!ok) failures++;
}

static uint64_t rng = 0x9e3779b97f4a7c15UL;
static float frand(void) {            /* roughly N(0,1): sum of 4 uniforms */
    float s = 0;
    for (int i = 0; i < 4; i++) {
        rng = rng * 6364136223846793005UL + 1442695040888963407UL;
        s += (float)(rng >> 40) / (float)(1UL << 24);
    }
    return (s - 2.0f) * 1.732f;
}

static float relerr(float a, float b) {
    float d = __builtin_fabsf(a - b), m = __builtin_fabsf(b);
    return d / (m > 1e-3f ? m : 1e-3f);
}

static void selftest_random(void) {
    static float x[4096], w[4096];
    static block_q4_0 wq[4096 / QK];
    static block_q8_0 xq[4096 / QK];
    static const int sizes[] = { 32, 64, 96, 1024, 4096 };
    float worst = 0;
    for (unsigned s = 0; s < ARRAY_LEN(sizes); s++) {
        int n = sizes[s];
        for (int i = 0; i < n; i++) { x[i] = frand(); w[i] = frand() * 0.05f; }
        quantize_row_q4_0(w, wq, n);
        quantize_row_q8_0(x, xq, n);
        float r = vec_dot_q4_0_q8_0_ref(n, wq, xq), v = vec_dot_q4_0_q8_0_neon(n, wq, xq);
        float e = relerr(v, r);
        if (e > worst) worst = e;
    }
    kprintf("  NEON vs scalar dot, 5 lengths 32..4096: worst rel err %.2e\n", (double)worst);
    check(worst < 1e-5f, "NEON Q4_0 x Q8_0 dot matches scalar reference");
}

static void test_vectors(void) {
    const testvec *tv = (const testvec *)BLOB_ADDR;
    if (memcmp(tv->magic, "Q4TV0001", 8) != 0) {
        kprintf("  no test-vector blob at %p, skipping\n", (void *)tv);
        failures++;
        return;
    }
    const char *base = (const char *)tv;
    int rows = (int)tv->rows, cols = (int)tv->cols, nb = cols / QK;
    const float *wf = (const float *)(base + tv->off_wf32);
    const block_q4_0 *wq = (const block_q4_0 *)(base + tv->off_wq);
    const float *x = (const float *)(base + tv->off_x);
    const block_q8_0 *xq = (const block_q8_0 *)(base + tv->off_xq);
    const float *yref = (const float *)(base + tv->off_yref);
    const float *yf32 = (const float *)(base + tv->off_yf32);
    kprintf("  blob: %d x %d, %u bytes\n", rows, cols, tv->total);
    if (rows > MAX_ROWS || (size_t)rows * nb > MAX_WQ || cols > 4096) { check(0, "blob fits buffers"); return; }

    /* 1. Our quantizers produce the same bytes as gguf-py's. */
    quantize_row_q4_0(wf, wq_dev, rows * cols);
    int q4_diff = 0;
    for (int b = 0; b < rows * nb; b++) q4_diff += memcmp(&wq_dev[b], &wq[b], sizeof(block_q4_0)) != 0;
    kprintf("  Q4_0 blocks differing from gguf-py: %d of %d\n", q4_diff, rows * nb);
    check(q4_diff == 0, "device Q4_0 quantizer is byte-identical to gguf-py");

    quantize_row_q8_0(x, xq_dev, cols);
    int q8_off = 0, q8_d = 0;
    for (int b = 0; b < nb; b++) {
        q8_d += xq_dev[b].d != xq[b].d;
        for (int j = 0; j < QK; j++) {
            int d = xq_dev[b].qs[j] - xq[b].qs[j];
            if (d > 1 || d < -1) q8_off += 100; else q8_off += d != 0;
        }
    }
    kprintf("  Q8_0: scale mismatches %d, values off by one %d (round-half ties)\n", q8_d, q8_off);
    check(q8_d == 0 && q8_off < 100, "device Q8_0 quantizer matches gguf-py within ties");

    /* 2. Matvec against the llama.cpp-formula reference, 1 core and all cores. */
    static const struct { int neon, cores; const char *name; } arms[] = {
        { 0, 1, "scalar, 1 core" }, { 1, 1, "NEON, 1 core" }, { 1, NCORES, "NEON, all cores" },
    };
    for (unsigned a = 0; a < ARRAY_LEN(arms); a++) {
        matvec_args m = { wq, xq, y_out, rows, cols, arms[a].neon };
        for (int r = 0; r < rows; r++) y_out[r] = -12345.0f;
        smp_run(matvec_q4_0_job, &m, arms[a].cores);
        float worst = 0;
        for (int r = 0; r < rows; r++) { float e = relerr(y_out[r], yref[r]); if (e > worst) worst = e; }
        kprintf("  matvec %-16s worst rel err vs reference %.2e\n", arms[a].name, (double)worst);
        check(worst < 1e-4f, "matvec matches llama.cpp-formula reference");
    }

    /* 3. Quantization error against the fp32 product (informational). */
    double num = 0, den = 0;
    for (int r = 0; r < rows; r++) { double d = y_out[r] - yf32[r]; num += d * d; den += (double)yf32[r] * yf32[r]; }
    kprintf("  Q4_0/Q8_0 vs fp32 matvec: relative RMS error %.4f (quantization, not a bug)\n",
            __builtin_sqrt(num / den));

    /* 4. Timing. Under QEMU TCG these are emulation speeds, not hardware. */
    double mb = (double)rows * nb * sizeof(block_q4_0) / 1e6;
    for (unsigned a = 0; a < ARRAY_LEN(arms); a++) {
        matvec_args m = { wq, xq, y_out, rows, cols, arms[a].neon };
        const int iters = 10;
        uint64_t t0 = ticks();
        for (int i = 0; i < iters; i++) smp_run(matvec_q4_0_job, &m, arms[a].cores);
        double s = seconds(ticks() - t0) / iters;
        kprintf("  time %-16s %.3f ms per matvec, %.1f MB/s of weights\n", arms[a].name, s * 1e3, mb / s);
    }
}

/* Qwen3 model mode: the blob at BLOB_ADDR is a model from tools/convert.py.
 * Buffers are static, sized for Qwen3-0.6B at a 512-token context: the KV
 * cache (f16) is 2 x 28 x 512 x 1024 x 2 bytes = 58.7 MB of bss. */
#define QW_MAX_LAYER 28
#define QW_MAX_DIM   1024
#define QW_MAX_QDIM  2048
#define QW_MAX_KV    1024
#define QW_MAX_FFN   3072
#define QW_MAX_CTX   512
#define QW_MAX_VOCAB 151936
static float m_x[QW_MAX_DIM], m_xn[QW_MAX_DIM], m_q[QW_MAX_QDIM], m_k[QW_MAX_KV], m_v[QW_MAX_KV],
             m_att[QW_MAX_QDIM], m_g[QW_MAX_FFN], m_u[QW_MAX_FFN], m_scores[QW_MAX_CTX], m_logits[QW_MAX_VOCAB];
static block_q8_0 m_xq[QW_MAX_FFN / QK];
static qw_kv m_kc[(size_t)QW_MAX_LAYER * QW_MAX_CTX * QW_MAX_KV], m_vc[(size_t)QW_MAX_LAYER * QW_MAX_CTX * QW_MAX_KV];

void qw_matvec(const block_q4_0 *w, const block_q8_0 *xq, const float *xf, float *y, int rows, int cols) {
    matvec_args m = { w, xq, y, rows, cols, 1 };
    smp_run(matvec_q4_0_job, &m, NCORES);
}

static void put_token(const qw_header *h, int id) {
    int n;
    const uint8_t *b = qw_token(h, id, &n);
    for (int i = 0; i < n; i++) uart_putc((char)b[i]);
}

static void run_model(const qw_header *h) {
    const char *err = qw_check(h);
    if (err) { kprintf("  model blob: %s\n", err); check(0, "model blob is usable"); return; }
    if (h->n_layer > QW_MAX_LAYER || h->dim > QW_MAX_DIM || h->n_head * h->head_dim > QW_MAX_QDIM ||
        h->n_kv * h->head_dim > QW_MAX_KV || h->ffn > QW_MAX_FFN || h->max_ctx > QW_MAX_CTX ||
        h->vocab > QW_MAX_VOCAB || h->n_expect > 64) {
        check(0, "model fits the kernel's static buffers");
        return;
    }
    kprintf("  Qwen3: %u layers, dim %u, %u/%u heads x %u, ffn %u, vocab %u, context %u, blob %u MB at %p\n",
            h->n_layer, h->dim, h->n_head, h->n_kv, h->head_dim, h->ffn, h->vocab, h->max_ctx,
            h->total >> 20, (void *)h);
    qw_state s = { m_x, m_xn, m_q, m_k, m_v, m_att, m_g, m_u, m_scores, m_logits, m_xq, m_kc, m_vc };
    const uint32_t *pr = qw_prompt(h), *ex = qw_expect(h), *dev = qw_expect_dev(h);
    int have_dev = 0;
    for (uint32_t i = 0; i < h->n_expect; i++) have_dev |= dev[i] != 0;

    kprintf("\n");
    uint64_t t0 = ticks();
    for (uint32_t i = 0; i < h->n_prompt; i++) { put_token(h, (int)pr[i]); qw_forward(h, &s, (int)pr[i], (int)i); }
    uint64_t t1 = ticks();
    int match_dev = 0, match_ref = 0, pos = (int)h->n_prompt;
    for (uint32_t i = 0; i < h->n_expect; i++) {
        int t = qw_argmax(s.logits, (int)h->vocab);
        match_dev += t == (int)dev[i];
        match_ref += t == (int)ex[i];
        put_token(h, t);
        if (i + 1 < h->n_expect) qw_forward(h, &s, t, pos++);
    }
    uint64_t t2 = ticks();
    double pf = seconds(t1 - t0) / h->n_prompt, dc = seconds(t2 - t1) / (h->n_expect - 1);
    kprintf("\n\n  prefill %.3f s/token, decode %.3f s/token (%.2f tok/s) on %d cores\n", pf, dc, 1.0 / dc, smp_cores());
    kprintf("  greedy tokens matching the PyTorch reference: %d of %u (it splits at a near-tie; see README)\n",
            match_ref, h->n_expect);
    if (!have_dev) { kprintf("  no host_run data in the blob: run `make check-host` first\n"); check(0, "blob has host_run check data"); return; }
    kprintf("  greedy tokens matching the host build (scalar dot products, same rounding otherwise): %d of %u\n",
            match_dev, h->n_expect);

    /* Teacher-forced logit check: feed prompt + PyTorch continuation and compare the
     * logits of host_run's top-k tokens at every position. Free-running tokens can
     * differ at a near-tie. The tolerance is the measured noise floor of this
     * arithmetic: two host builds differing only in FMA contraction disagree by up
     * to 0.57 logits (results/host_check.json), because each Q8_0 re-quantization
     * can turn a 1-ulp difference into a rounding flip. A real bug (RoPE, head
     * mapping, norms) moves logits by whole units and garbles the text. */
    const qw_check_entry *ce = qw_checks(h);
    int npos = (int)(h->n_prompt + h->n_expect), k = (int)h->check_k, top1 = 0;
    float worst = 0;
    for (int p = 0; p < npos; p++) {
        int t = p < (int)h->n_prompt ? (int)pr[p] : (int)ex[p - h->n_prompt];
        qw_forward(h, &s, t, p);
        top1 += qw_argmax(s.logits, (int)h->vocab) == (int)ce[p * k].id;
        for (int j = 0; j < k; j++) {
            float d = __builtin_fabsf(s.logits[ce[p * k + j].id] - ce[p * k + j].logit);
            if (d > worst) worst = d;
        }
    }
    kprintf("  teacher-forced over %d positions: top-1 same as host build at %d, worst top-%d logit difference %.2e\n",
            npos, top1, k, (double)worst);
    check(worst < 1.0f && top1 >= npos - 2, "kernel logits match the host build within the rounding noise floor (< 1.0)");
}

/* Streaming read bandwidth: the ceiling on decode speed, since every token
 * reads every weight once. */
typedef struct { const uint8_t *buf; size_t bytes; uint64_t sink[NCORES]; } bw_args;
static void bw_job(int core, int ncores, void *arg) {
    bw_args *a = arg;
    size_t per = a->bytes / ncores;
    const uint8_t *p = a->buf + core * per, *end = p + per;
    uint8x16_t s = vdupq_n_u8(0);
    for (; p < end; p += 64) {
        uint8x16x4_t v = vld1q_u8_x4(p);
        s = veorq_u8(s, veorq_u8(veorq_u8(v.val[0], v.val[1]), veorq_u8(v.val[2], v.val[3])));
    }
    a->sink[core] = vgetq_lane_u64(vreinterpretq_u64_u8(s), 0);
}

static void bandwidth(void) {
    const size_t bytes = 64UL << 20;
    bw_args a = { (const uint8_t *)(uintptr_t)BW_ADDR, bytes, {0} };
    for (int cores = 1; cores <= smp_cores(); cores *= 2) {
        smp_run(bw_job, &a, cores);                  /* warm */
        uint64_t t0 = ticks();
        smp_run(bw_job, &a, cores);
        double gbs = (double)bytes / seconds(ticks() - t0) / 1e9;
        /* Qwen3-0.6B at Q4_0: ~596M params x 4.5 bits = ~0.335 GB read per token
         * (tied embeddings: the output layer reads the whole 151,936 x 1024 table). */
        kprintf("  read %d core%s: %.2f GB/s -> decode ceiling ~%.1f tok/s for Qwen3-0.6B Q4_0\n",
                cores, cores > 1 ? "s" : " ", gbs, gbs / 0.335);
    }
}

void kmain(void) {
    sysreg_write(vbar_el1, (uint64_t)vectors);
    isb();
    uint64_t el = (sysreg_read(CurrentEL) >> 2) & 3;
    kprintf("\ncarthing-llm-os on %s: booted at EL%lu, running at EL%lu, timer %lu Hz\n",
            PLAT_NAME, boot_el, el, tick_hz());
    mmu_build();
    mmu_enable();
    kprintf("MMU and caches on (SCTLR_EL1=%lx)\n", sysreg_read(sctlr_el1));
    uint32_t v = psci_version();
    kprintf("PSCI %u.%u via %s\n", v >> 16, v & 0xffff, boot_el >= 2 || !PSCI_AUTO ? "smc" : "hvc");
    int n = smp_init();
    kprintf("cores online: %d of %d\n", n, NCORES);
    check(n == NCORES, "all cores started through PSCI CPU_ON");

    kprintf("self-test:\n");
    selftest_random();
    if (qw_check((const void *)BLOB_ADDR) == 0 || memcmp((const void *)BLOB_ADDR, QW_MAGIC, 8) == 0) {
        kprintf("model:\n");
        run_model((const qw_header *)BLOB_ADDR);
    } else {
        kprintf("test vectors:\n");
        test_vectors();
    }
    kprintf("memory bandwidth:\n");
    bandwidth();

    if (failures) kprintf("RESULT: FAIL (%d)\n", failures);
    else kprintf("RESULT: PASS\n");
    psci_system_off();
}
