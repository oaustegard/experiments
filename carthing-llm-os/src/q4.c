#include "q4.h"
#include <arm_neon.h>

static inline float absf(float v) { return __builtin_fabsf(v); }

/* Same arithmetic as ggml's quantize_row_q4_0_ref. FMA contraction is off:
 * a fused x*id + 8.5 rounds differently and changed 1 block in 32,000. */
void quantize_row_q4_0(const float *x, block_q4_0 *y, int n) {
#pragma clang fp contract(off)
    for (int i = 0; i < n / QK; i++, x += QK) {
        float amax = 0, max = 0;
        for (int j = 0; j < QK; j++)
            if (amax < absf(x[j])) { amax = absf(x[j]); max = x[j]; }
        float d = max / -8.0f, id = d ? 1.0f / d : 0.0f;
        y[i].d = f32_to_f16(d);
        for (int j = 0; j < QK / 2; j++) {
            int q0 = (int8_t)(x[j] * id + 8.5f), q1 = (int8_t)(x[j + QK / 2] * id + 8.5f);
            if (q0 > 15) q0 = 15;
            if (q1 > 15) q1 = 15;
            y[i].qs[j] = (uint8_t)(q0 | (q1 << 4));
        }
    }
}

/* Same arithmetic as ggml's quantize_row_q8_0_ref. */
void quantize_row_q8_0(const float *x, block_q8_0 *y, int n) {
#pragma clang fp contract(off)
    for (int i = 0; i < n / QK; i++, x += QK) {
        float amax = 0;
        for (int j = 0; j < QK; j++) if (amax < absf(x[j])) amax = absf(x[j]);
        float d = amax / 127.0f, id = d ? 1.0f / d : 0.0f;
        y[i].d = f32_to_f16(d);
        for (int j = 0; j < QK; j++) y[i].qs[j] = (int8_t)__builtin_roundf(x[j] * id);
    }
}

/* Low nibble of qs[j] is element j, high nibble is element j+16. */
float vec_dot_q4_0_q8_0_ref(int n, const block_q4_0 *x, const block_q8_0 *y) {
    float sum = 0;
    for (int i = 0; i < n / QK; i++) {
        int s = 0;
        for (int j = 0; j < QK / 2; j++) {
            s += ((x[i].qs[j] & 0x0f) - 8) * y[i].qs[j];
            s += ((x[i].qs[j] >> 4) - 8) * y[i].qs[j + QK / 2];
        }
        sum += (float)s * f16_to_f32(x[i].d) * f16_to_f32(y[i].d);
    }
    return sum;
}

/* Cortex-A53 is ARMv8.0: no sdot. Widen with smull, pairwise-add to int32,
 * then scale each block's integer sum by d_x * d_y in float. Two blocks per
 * iteration to keep two accumulators in flight. */
float vec_dot_q4_0_q8_0_neon(int n, const block_q4_0 *x, const block_q8_0 *y) {
    const int nb = n / QK;
    const uint8x16_t m4 = vdupq_n_u8(0x0f);
    const int8x16_t s8 = vdupq_n_s8(8);
    float32x4_t acc0 = vdupq_n_f32(0), acc1 = vdupq_n_f32(0);
    int i = 0;
    for (; i + 1 < nb; i += 2) {
        const block_q4_0 *x0 = &x[i], *x1 = &x[i + 1];
        const block_q8_0 *y0 = &y[i], *y1 = &y[i + 1];
        uint8x16_t b0 = vld1q_u8(x0->qs), b1 = vld1q_u8(x1->qs);
        int8x16_t l0 = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(b0, m4)), s8);
        int8x16_t h0 = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(b0, 4)), s8);
        int8x16_t l1 = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(b1, m4)), s8);
        int8x16_t h1 = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(b1, 4)), s8);
        int8x16_t ya0 = vld1q_s8(y0->qs), yb0 = vld1q_s8(y0->qs + 16);
        int8x16_t ya1 = vld1q_s8(y1->qs), yb1 = vld1q_s8(y1->qs + 16);

        int16x8_t p;
        int32x4_t s0, s1;
        p  = vmull_s8(vget_low_s8(l0), vget_low_s8(ya0));   s0 = vpaddlq_s16(p);
        p  = vmull_s8(vget_high_s8(l0), vget_high_s8(ya0)); s0 = vpadalq_s16(s0, p);
        p  = vmull_s8(vget_low_s8(h0), vget_low_s8(yb0));   s0 = vpadalq_s16(s0, p);
        p  = vmull_s8(vget_high_s8(h0), vget_high_s8(yb0)); s0 = vpadalq_s16(s0, p);
        p  = vmull_s8(vget_low_s8(l1), vget_low_s8(ya1));   s1 = vpaddlq_s16(p);
        p  = vmull_s8(vget_high_s8(l1), vget_high_s8(ya1)); s1 = vpadalq_s16(s1, p);
        p  = vmull_s8(vget_low_s8(h1), vget_low_s8(yb1));   s1 = vpadalq_s16(s1, p);
        p  = vmull_s8(vget_high_s8(h1), vget_high_s8(yb1)); s1 = vpadalq_s16(s1, p);

        acc0 = vmlaq_n_f32(acc0, vcvtq_f32_s32(s0), f16_to_f32(x0->d) * f16_to_f32(y0->d));
        acc1 = vmlaq_n_f32(acc1, vcvtq_f32_s32(s1), f16_to_f32(x1->d) * f16_to_f32(y1->d));
    }
    float sum = vaddvq_f32(vaddq_f32(acc0, acc1));
    if (i < nb) sum += vec_dot_q4_0_q8_0_ref(QK, &x[i], &y[i]);
    return sum;
}

void matvec_q4_0_job(int core, int ncores, void *arg) {
    matvec_args *a = arg;
    int per = (a->rows + ncores - 1) / ncores;
    int r0 = core * per, r1 = r0 + per;
    if (r1 > a->rows) r1 = a->rows;
    int nb = a->cols / QK;
    for (int r = r0; r < r1; r++)
        a->y[r] = a->neon ? vec_dot_q4_0_q8_0_neon(a->cols, a->w + (size_t)r * nb, a->x)
                          : vec_dot_q4_0_q8_0_ref(a->cols, a->w + (size_t)r * nb, a->x);
}
