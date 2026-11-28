/* Qwen3 forward pass over a flat model blob written by tools/convert.py.
 * Portable C with no libc: the kernel and tools/host_run.c build the same file.
 * The caller supplies the matrix-vector product (qw_matvec) and the buffers. */
#pragma once
#include "q4.h"

#define QW_MAGIC "QW3M0001"

typedef struct {
    char magic[8];
    uint32_t n_layer, dim, n_head, n_kv, head_dim, ffn, vocab, max_ctx;
    float rms_eps;
    uint32_t n_prompt, n_expect;      /* prompt token ids; greedy continuation from the PyTorch reference */
    uint32_t off_tok_embd;            /* Q4_0 [vocab][dim], also the output layer (tied) */
    uint32_t off_out_norm;            /* f32 [dim] */
    uint32_t off_rope;                /* f32 [max_ctx][head_dim/2][2]: cos, sin */
    uint32_t off_vocab_idx;           /* u32 [vocab+1]: byte offsets into the token-bytes table */
    uint32_t off_vocab_bytes;
    uint32_t off_prompt, off_expect;  /* u32 token ids */
    uint32_t off_expect_dev;          /* u32 [n_expect]: greedy tokens from host_run, the device's arithmetic */
    uint32_t off_check, check_k;      /* qw_check_entry [n_prompt + n_expect][check_k]: host_run's top-k logits,
                                         teacher-forced over prompt + PyTorch continuation */
    uint32_t off_layers;              /* qw_layer_off [n_layer] */
    uint32_t total;
} qw_header;

typedef struct {
    uint32_t attn_norm, wq, wk, wv, q_norm, k_norm, wo, ffn_norm, w_gate, w_up, w_down;
} qw_layer_off;

typedef struct { uint32_t id; float logit; } qw_check_entry;

/* QW_EXACT (host only): f32 KV cache, and qw_matvec is handed the float input
 * too, so a build can drop both of the device's approximations and isolate
 * bugs from rounding. */
#ifdef QW_EXACT
typedef float qw_kv;
#else
typedef _Float16 qw_kv;
#endif

/* Activations and KV cache; the caller allocates. */
typedef struct {
    float *x, *xn, *q, *k, *v, *att, *g, *u, *scores, *logits;
    block_q8_0 *xq;
    qw_kv *kcache, *vcache;           /* [n_layer][max_ctx][n_kv * head_dim] */
} qw_state;

/* Provided by the platform: y[r] = W[r,:] . x for r < rows. xq is x quantized to
 * Q8_0 (what the device uses); xf is the float x (used only by QW_EXACT builds). */
void qw_matvec(const block_q4_0 *w, const block_q8_0 *xq, const float *xf, float *y, int rows, int cols);

const char *qw_check(const void *blob);   /* NULL if usable, else what is wrong */
void qw_forward(const qw_header *h, qw_state *s, int token, int pos);   /* fills s->logits */
int qw_argmax(const float *v, int n);
const uint8_t *qw_token(const qw_header *h, int id, int *len);
static inline const uint32_t *qw_prompt(const qw_header *h) { return (const uint32_t *)((const char *)h + h->off_prompt); }
static inline const uint32_t *qw_expect(const qw_header *h) { return (const uint32_t *)((const char *)h + h->off_expect); }
static inline const uint32_t *qw_expect_dev(const qw_header *h) { return (const uint32_t *)((const char *)h + h->off_expect_dev); }
static inline const qw_check_entry *qw_checks(const qw_header *h) { return (const qw_check_entry *)((const char *)h + h->off_check); }
float qw_expf(float x);
