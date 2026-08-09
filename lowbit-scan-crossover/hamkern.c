// Sign-bit Hamming scan, k=256 (4x uint64 per vector), single-threaded.
// Built with -O3 -march=native so gcc can use VPOPCNTDQ where available.
#include <stdint.h>
#include <stddef.h>

void ham_scan(const uint64_t *restrict C, const uint64_t *restrict q,
              uint16_t *restrict out, long n) {
    const uint64_t q0 = q[0], q1 = q[1], q2 = q[2], q3 = q[3];
    for (long i = 0; i < n; i++) {
        const uint64_t *r = C + i * 4;
        out[i] = (uint16_t)(__builtin_popcountll(r[0] ^ q0) +
                            __builtin_popcountll(r[1] ^ q1) +
                            __builtin_popcountll(r[2] ^ q2) +
                            __builtin_popcountll(r[3] ^ q3));
    }
}

// Fused scan + top-k-by-threshold: emit candidate ids with distance <= thr.
// Avoids materialising a full score array at all.
long ham_scan_thresh(const uint64_t *restrict C, const uint64_t *restrict q,
                     uint32_t *restrict ids, long n, int thr, long cap) {
    const uint64_t q0 = q[0], q1 = q[1], q2 = q[2], q3 = q[3];
    long m = 0;
    for (long i = 0; i < n; i++) {
        const uint64_t *r = C + i * 4;
        int d = __builtin_popcountll(r[0] ^ q0) + __builtin_popcountll(r[1] ^ q1) +
                __builtin_popcountll(r[2] ^ q2) + __builtin_popcountll(r[3] ^ q3);
        if (d <= thr) { if (m < cap) ids[m] = (uint32_t)i; m++; }
    }
    return m;
}

// Pure streaming-read bandwidth probe over the same buffer.
uint64_t band_probe(const uint64_t *restrict C, long words) {
    uint64_t acc = 0;
    for (long i = 0; i < words; i++) acc ^= C[i];
    return acc;
}
