#include "qwen3.h"

#define AT(h, off, type) ((const type *)((const char *)(h) + (off)))

/* expf without libm: e^x = 2^k * e^r, |r| <= ln2/2, degree-6 Taylor for e^r
 * (relative error about 1.3e-7). Same code on host and device. */
float qw_expf(float x) {
    if (x < -87.0f) return 0.0f;
    if (x > 88.0f) x = 88.0f;
    float k = __builtin_roundf(x * 1.44269504f);
    float r = x - k * 0.693145751953125f - k * 1.428606765330187e-06f;
    float p = 1.0f + r * (1.0f + r * (0.5f + r * (0.16666667f + r * (0.041666668f
              + r * (0.008333334f + r * 0.0013888889f)))));
    union { uint32_t u; float f; } s = { (uint32_t)((int)k + 127) << 23 };
    return p * s.f;
}

const char *qw_check(const void *blob) {
    const qw_header *h = blob;
    const char *m = QW_MAGIC;
    for (int i = 0; i < 8; i++) if (h->magic[i] != m[i]) return "bad magic";
    if (h->dim % QK || h->ffn % QK || (h->n_head * h->head_dim) % QK) return "dims not multiples of 32";
    if (h->n_head % h->n_kv) return "n_head not a multiple of n_kv";
    if (h->head_dim % 2) return "odd head_dim";
    if (h->n_prompt == 0 || h->n_prompt + h->n_expect > h->max_ctx) return "prompt does not fit the context";
    return 0;
}

static void rmsnorm(float *out, const float *x, const float *w, int n, float eps) {
    float ss = 0;
    for (int i = 0; i < n; i++) ss += x[i] * x[i];
    float inv = 1.0f / __builtin_sqrtf(ss / (float)n + eps);
    for (int i = 0; i < n; i++) out[i] = x[i] * inv * w[i];
}

/* Rotate-half RoPE (the HF/NeoX layout): element i pairs with i + hd/2. */
static void rope(float *v, const float *cs, int hd) {
    int half = hd / 2;
    for (int i = 0; i < half; i++) {
        float c = cs[2 * i], s = cs[2 * i + 1];
        float a = v[i], b = v[i + half];
        v[i] = a * c - b * s;
        v[i + half] = b * c + a * s;
    }
}

static void matvec(const qw_header *h, uint32_t off, qw_state *s, const float *x, float *y, int rows, int cols) {
    quantize_row_q8_0(x, s->xq, cols);
    qw_matvec(AT(h, off, block_q4_0), s->xq, x, y, rows, cols);
}

void qw_forward(const qw_header *h, qw_state *s, int token, int pos) {
    const int dim = h->dim, hd = h->head_dim, nh = h->n_head, nkv = h->n_kv, ffn = h->ffn;
    const int qdim = nh * hd, kvdim = nkv * hd, group = nh / nkv;
    const float eps = h->rms_eps, scale = 1.0f / __builtin_sqrtf((float)hd);
    const float *cs = AT(h, h->off_rope, float) + (size_t)pos * hd;   /* hd/2 pairs */
    const qw_layer_off *L = AT(h, h->off_layers, qw_layer_off);

    dequantize_row_q4_0(AT(h, h->off_tok_embd, block_q4_0) + (size_t)token * (dim / QK), s->x, dim);

    for (uint32_t l = 0; l < h->n_layer; l++) {
        const qw_layer_off *lo = &L[l];
        qw_kv *kc = s->kcache + ((size_t)l * h->max_ctx) * kvdim;
        qw_kv *vc = s->vcache + ((size_t)l * h->max_ctx) * kvdim;

        rmsnorm(s->xn, s->x, AT(h, lo->attn_norm, float), dim, eps);
        quantize_row_q8_0(s->xn, s->xq, dim);              /* one quantization feeds q, k and v */
        qw_matvec(AT(h, lo->wq, block_q4_0), s->xq, s->xn, s->q, qdim, dim);
        qw_matvec(AT(h, lo->wk, block_q4_0), s->xq, s->xn, s->k, kvdim, dim);
        qw_matvec(AT(h, lo->wv, block_q4_0), s->xq, s->xn, s->v, kvdim, dim);

        for (int i = 0; i < nh; i++) {
            rmsnorm(s->q + i * hd, s->q + i * hd, AT(h, lo->q_norm, float), hd, eps);
            rope(s->q + i * hd, cs, hd);
        }
        for (int i = 0; i < nkv; i++) {
            rmsnorm(s->k + i * hd, s->k + i * hd, AT(h, lo->k_norm, float), hd, eps);
            rope(s->k + i * hd, cs, hd);
        }
        for (int i = 0; i < kvdim; i++) {
            kc[(size_t)pos * kvdim + i] = (qw_kv)s->k[i];
            vc[(size_t)pos * kvdim + i] = (qw_kv)s->v[i];
        }

        for (int hq = 0; hq < nh; hq++) {
            const float *q = s->q + hq * hd;
            const int kvh = hq / group;
            float mx = -3.0e38f;
            for (int t = 0; t <= pos; t++) {
                const qw_kv *k = kc + (size_t)t * kvdim + kvh * hd;
                float d = 0;
                for (int i = 0; i < hd; i++) d += q[i] * (float)k[i];
                s->scores[t] = d * scale;
                if (s->scores[t] > mx) mx = s->scores[t];
            }
            float sum = 0;
            for (int t = 0; t <= pos; t++) { s->scores[t] = qw_expf(s->scores[t] - mx); sum += s->scores[t]; }
            float *o = s->att + hq * hd;
            for (int i = 0; i < hd; i++) o[i] = 0;
            for (int t = 0; t <= pos; t++) {
                const qw_kv *v = vc + (size_t)t * kvdim + kvh * hd;
                float p = s->scores[t] / sum;
                for (int i = 0; i < hd; i++) o[i] += p * (float)v[i];
            }
        }
        matvec(h, lo->wo, s, s->att, s->xn, dim, qdim);
        for (int i = 0; i < dim; i++) s->x[i] += s->xn[i];

        rmsnorm(s->xn, s->x, AT(h, lo->ffn_norm, float), dim, eps);
        quantize_row_q8_0(s->xn, s->xq, dim);
        qw_matvec(AT(h, lo->w_gate, block_q4_0), s->xq, s->xn, s->g, ffn, dim);
        qw_matvec(AT(h, lo->w_up, block_q4_0), s->xq, s->xn, s->u, ffn, dim);
        for (int i = 0; i < ffn; i++) s->g[i] = s->g[i] / (1.0f + qw_expf(-s->g[i])) * s->u[i];
        matvec(h, lo->w_down, s, s->g, s->xn, dim, ffn);
        for (int i = 0; i < dim; i++) s->x[i] += s->xn[i];
    }

    rmsnorm(s->xn, s->x, AT(h, h->off_out_norm, float), dim, eps);
    matvec(h, h->off_tok_embd, s, s->xn, s->logits, h->vocab, dim);
}

int qw_argmax(const float *v, int n) {
    int best = 0;
    for (int i = 1; i < n; i++) if (v[i] > v[best]) best = i;
    return best;
}

const uint8_t *qw_token(const qw_header *h, int id, int *len) {
    const uint32_t *idx = AT(h, h->off_vocab_idx, uint32_t);
    *len = (int)(idx[id + 1] - idx[id]);
    return AT(h, h->off_vocab_bytes, uint8_t) + idx[id];
}
