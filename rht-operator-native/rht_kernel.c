#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/*
 * Structured randomized Hadamard rotation, same plan as
 * remex.rotation.rht_rotation (rounds of permute -> sign -> block FWHT).
 *
 * Every step is an elementwise IEEE operation in a fixed order (gather,
 * multiply, pairwise add/sub), so the output is bit-identical across
 * compilers, instruction sets and thread counts. Build with
 * -ffp-contract=off so no multiply is ever fused into an add.
 */

static inline void fwht_block(float *x, int64_t B) {
    for (int64_t h = 1; h < B; h <<= 1)
        for (int64_t i = 0; i < B; i += 2 * h)
            for (int64_t j = i; j < i + h; j++) {
                float a = x[j], b = x[j + h];
                x[j] = a + b;
                x[j + h] = a - b;
            }
}

static inline void fwht_all(float *x, int64_t d, int64_t B) {
    for (int64_t o = 0; o < d; o += B) fwht_block(x + o, B);
}

/* mode 0: dst = src @ R   (decode)          mode 1: dst = src @ R.T (encode, query) */
static void apply_row(const float *src, float *dst, float *buf, int64_t d,
                      int64_t B, int64_t rounds, const int32_t *perms,
                      const float *ss, int32_t mode) {
    if (mode == 0) {
        for (int64_t r = 0; r < rounds; r++) {
            const int32_t *p = perms + r * d;
            const float *s = ss + r * d;
            const float *in = src;
            if (r > 0) { memcpy(buf, dst, (size_t)d * sizeof(float)); in = buf; }
            for (int64_t j = 0; j < d; j++) dst[j] = in[p[j]] * s[j];
            fwht_all(dst, d, B);
        }
    } else {
        memcpy(dst, src, (size_t)d * sizeof(float));
        for (int64_t r = rounds - 1; r >= 0; r--) {
            const int32_t *p = perms + r * d;
            const float *s = ss + r * d;
            fwht_all(dst, d, B);
            memcpy(buf, dst, (size_t)d * sizeof(float));
            for (int64_t j = 0; j < d; j++) dst[p[j]] = buf[j] * s[j];
        }
    }
}

int rht_apply(const float *X, float *out, int64_t n, int64_t d, int64_t B,
              int64_t rounds, const int32_t *perms, const float *ss,
              int32_t mode, int32_t nthreads) {
    int err = 0;
    if (nthreads <= 1 || n <= 1) {
        float *buf = (float *)malloc((size_t)d * sizeof(float));
        if (!buf) return 1;
        for (int64_t i = 0; i < n; i++)
            apply_row(X + i * d, out + i * d, buf, d, B, rounds, perms, ss, mode);
        free(buf);
        return 0;
    }
    #pragma omp parallel num_threads(nthreads) 
    {
        float *buf = (float *)malloc((size_t)d * sizeof(float));
        if (!buf) {
            #pragma omp atomic write
            err = 1;
        } else {
            #pragma omp for schedule(static)
            for (int64_t i = 0; i < n; i++)
                apply_row(X + i * d, out + i * d, buf, d, B, rounds, perms, ss, mode);
            free(buf);
        }
    }
    return err;
}
