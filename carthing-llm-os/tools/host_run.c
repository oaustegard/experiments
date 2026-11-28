/* Runs src/qwen3.c natively on the host, with the scalar Q4_0 dot product
 * split over threads, so the forward pass can be checked in seconds rather
 * than in QEMU.
 *
 *   host_run BLOB gen [--write-expect-dev]   greedy from the blob's prompt; compare with the PyTorch
 *                                     tokens; optionally store its own tokens in the blob for the kernel to match
 *   host_run BLOB logits IDS OUT      teacher-forced over u32 token ids; write f32 logits [n][vocab]
 */
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "qwen3.h"

#define NTHREADS 4

typedef struct { const block_q4_0 *w; const block_q8_0 *x; const float *xf; float *y; int r0, r1, cols; } part;
static void *run_part(void *arg) {
    part *p = arg;
    int nb = p->cols / QK;
#ifdef QW_EXACT
    float row[8192];
    for (int r = p->r0; r < p->r1; r++) {
        dequantize_row_q4_0(p->w + (size_t)r * nb, row, p->cols);
        double acc = 0;
        for (int i = 0; i < p->cols; i++) acc += (double)row[i] * p->xf[i];
        p->y[r] = (float)acc;
    }
#else
    for (int r = p->r0; r < p->r1; r++) p->y[r] = vec_dot_q4_0_q8_0_ref(p->cols, p->w + (size_t)r * nb, p->x);
#endif
    return NULL;
}

void qw_matvec(const block_q4_0 *w, const block_q8_0 *x, const float *xf, float *y, int rows, int cols) {
    pthread_t th[NTHREADS];
    part p[NTHREADS];
    int per = (rows + NTHREADS - 1) / NTHREADS;
    for (int i = 0; i < NTHREADS; i++) {
        p[i] = (part){ w, x, xf, y, i * per, (i + 1) * per > rows ? rows : (i + 1) * per, cols };
        pthread_create(&th[i], NULL, run_part, &p[i]);
    }
    for (int i = 0; i < NTHREADS; i++) pthread_join(th[i], NULL);
}

static void *slurp(const char *path, size_t *n) {
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); exit(1); }
    fseek(f, 0, SEEK_END);
    *n = (size_t)ftell(f);
    fseek(f, 0, SEEK_SET);
    void *buf = aligned_alloc(64, (*n + 63) / 64 * 64);
    if (fread(buf, 1, *n, f) != *n) { perror("read"); exit(1); }
    fclose(f);
    return buf;
}

static qw_state alloc_state(const qw_header *h) {
    int kvdim = h->n_kv * h->head_dim, qdim = h->n_head * h->head_dim;
    int maxin = (int)h->ffn > qdim ? (int)h->ffn : qdim;
    qw_state s = {
        .x = calloc(h->dim, 4), .xn = calloc(h->dim, 4), .q = calloc(qdim, 4),
        .k = calloc(kvdim, 4), .v = calloc(kvdim, 4), .att = calloc(qdim, 4),
        .g = calloc(h->ffn, 4), .u = calloc(h->ffn, 4), .scores = calloc(h->max_ctx, 4),
        .logits = calloc(h->vocab, 4), .xq = calloc(maxin / QK, sizeof(block_q8_0)),
        .kcache = calloc((size_t)h->n_layer * h->max_ctx * kvdim, sizeof(qw_kv)),
        .vcache = calloc((size_t)h->n_layer * h->max_ctx * kvdim, sizeof(qw_kv)),
    };
    return s;
}

static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }

static void put_token(const qw_header *h, int id) {
    int n;
    const uint8_t *b = qw_token(h, id, &n);
    fwrite(b, 1, n, stdout);
    fflush(stdout);
}

int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s BLOB gen | BLOB logits IDS OUT\n", argv[0]); return 2; }
    size_t n;
    const qw_header *h = slurp(argv[1], &n);
    const char *err = qw_check(h);
    if (err) { fprintf(stderr, "blob: %s\n", err); return 1; }
    qw_state s = alloc_state(h);

    if (!strcmp(argv[2], "gen")) {
        const uint32_t *pr = qw_prompt(h), *ex = qw_expect(h);
        double t0 = now();
        for (uint32_t i = 0; i < h->n_prompt; i++) { put_token(h, pr[i]); qw_forward(h, &s, pr[i], i); }
        double t1 = now();
        int match = 0, pos = h->n_prompt;
        uint32_t got[64];
        if (h->n_expect > 64) return 1;
        for (uint32_t i = 0; i < h->n_expect; i++) {
            int t = qw_argmax(s.logits, h->vocab);
            got[i] = (uint32_t)t;
            match += t == (int)ex[i];
            put_token(h, t);
            if (i + 1 < h->n_expect) qw_forward(h, &s, t, pos++);
        }
        double t2 = now();
        printf("\nprefill %.2f s/token, decode %.2f s/token, greedy tokens matching PyTorch: %d of %u\n",
               (t1 - t0) / h->n_prompt, (t2 - t1) / (h->n_expect - 1), match, h->n_expect);
        if (argc > 3 && !strcmp(argv[3], "--write-expect-dev")) {
            FILE *f = fopen(argv[1], "r+b");
            if (!f) { perror(argv[1]); return 1; }
            fseek(f, h->off_expect_dev, SEEK_SET);
            fwrite(got, 4, h->n_expect, f);
            fclose(f);
            printf("wrote these %u tokens to the blob as the kernel's expected output\n", h->n_expect);
            /* Teacher-forced top-k logits over prompt + PyTorch continuation, for the kernel's logit check. */
            int npos = (int)(h->n_prompt + h->n_expect), k = (int)h->check_k;
            qw_check_entry *ce = calloc((size_t)npos * k, sizeof *ce);
            for (int p = 0; p < npos; p++) {
                int t = p < (int)h->n_prompt ? (int)pr[p] : (int)ex[p - h->n_prompt];
                qw_forward(h, &s, t, p);
                for (int j = 0; j < k; j++) {          /* top-k by repeated scan; k is small */
                    int best = -1;
                    for (int v = 0; v < (int)h->vocab; v++) {
                        int taken = 0;
                        for (int q = 0; q < j; q++) taken |= ce[p * k + q].id == (uint32_t)v;
                        if (!taken && (best < 0 || s.logits[v] > s.logits[best])) best = v;
                    }
                    ce[p * k + j] = (qw_check_entry){ (uint32_t)best, s.logits[best] };
                }
            }
            f = fopen(argv[1], "r+b");
            fseek(f, h->off_check, SEEK_SET);
            fwrite(ce, sizeof *ce, (size_t)npos * k, f);
            fclose(f);
            printf("wrote top-%d logits at %d teacher-forced positions for the kernel's logit check\n", k, npos);
        }
        return 0;
    }
    if (!strcmp(argv[2], "logits") && argc == 5) {
        size_t nb;
        const uint32_t *ids = slurp(argv[3], &nb);
        int nt = (int)(nb / 4);
        if (nt > (int)h->max_ctx) { fprintf(stderr, "sequence longer than max_ctx\n"); return 1; }
        FILE *out = fopen(argv[4], "wb");
        for (int i = 0; i < nt; i++) {
            qw_forward(h, &s, ids[i], i);
            fwrite(s.logits, 4, h->vocab, out);
        }
        fclose(out);
        return 0;
    }
    fprintf(stderr, "unknown mode\n");
    return 2;
}
