/* llama.cpp's Q4_0 and Q8_0 block formats, bit-compatible with GGUF files.
 * Portable: also built natively on the host by tools/host_run.c. */
#pragma once
#include <stdint.h>
#include <stddef.h>

#define QK 32
typedef struct { uint16_t d; uint8_t qs[QK / 2]; } block_q4_0;   /* 18 bytes: x = (q - 8) * d */
typedef struct { uint16_t d; int8_t  qs[QK]; }     block_q8_0;   /* 34 bytes: x = q * d */
_Static_assert(sizeof(block_q4_0) == 18, "q4_0 layout");
_Static_assert(sizeof(block_q8_0) == 34, "q8_0 layout");

static inline float f16_to_f32(uint16_t h) { union { uint16_t u; _Float16 f; } v = { h }; return (float)v.f; }
static inline uint16_t f32_to_f16(float f) { union { _Float16 f; uint16_t u; } v = { (_Float16)f }; return v.u; }

void quantize_row_q4_0(const float *x, block_q4_0 *y, int n);
void quantize_row_q8_0(const float *x, block_q8_0 *y, int n);
float vec_dot_q4_0_q8_0_ref(int n, const block_q4_0 *x, const block_q8_0 *y);
float vec_dot_q4_0_q8_0_neon(int n, const block_q4_0 *x, const block_q8_0 *y);   /* AArch64 only */
void dequantize_row_q4_0(const block_q4_0 *x, float *y, int n);

/* y[r] = W[r,:] . x for rows [0, rows), split across cores by row. */
typedef struct {
    const block_q4_0 *w;
    const block_q8_0 *x;
    float *y;
    int rows, cols, neon;
} matvec_args;
void matvec_q4_0_job(int core, int ncores, void *arg);
