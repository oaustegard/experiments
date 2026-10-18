/* Dictwords, hand-translated from kernels/dictwords.py.
 * String counting in an open-addressing hash table (FNV-1a, linear probing),
 * then sort by (-count, word) with byte-wise word compare, which is Python's
 * tuple order for ASCII strings. All LCG values are non-negative, so C's
 * % and / match Python's floor semantics here. */
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define DEFAULT_N 500000
#define TBITS 16
#define TSIZE (1u << TBITS)
#define TMASK (TSIZE - 1)

static const char *SYLLABLES[12] = {"ka", "ri", "mo", "tu", "le", "sa",
                                    "no", "vi", "da", "pe", "zo", "hu"};

typedef struct {
    char w[16];      /* at most 8 chars for k <= 4 syllables; NUL-terminated */
    int len;
    long long count;
    int used;
} Entry;

static Entry table[TSIZE];
static Entry *uniq[TSIZE];
static size_t nuniq = 0;

static int make_word(long long x, char *out) {
    int len = 0;
    long long k = 1 + x % 4;
    for (long long t = 0; t < k; t++) {
        const char *s = SYLLABLES[x % 12];
        out[len++] = s[0];
        out[len++] = s[1];
        x = (x * 1103515245LL + 12345LL) % 2147483648LL;
        x = x / 7;
    }
    out[len] = '\0';
    return len;
}

static uint32_t fnv1a(const char *s, int len) {
    uint32_t h = 2166136261u;
    for (int i = 0; i < len; i++) {
        h ^= (unsigned char)s[i];
        h *= 16777619u;
    }
    return h;
}

static void count_word(const char *w, int len) {
    uint32_t idx = fnv1a(w, len) & TMASK;
    for (;;) {
        Entry *e = &table[idx];
        if (!e->used) {
            e->used = 1;
            memcpy(e->w, w, (size_t)len + 1);
            e->len = len;
            e->count = 1;
            uniq[nuniq++] = e;
            return;
        }
        if (e->len == len && memcmp(e->w, w, (size_t)len) == 0) {
            e->count++;
            return;
        }
        idx = (idx + 1) & TMASK;
    }
}

static int cmp_entry(const void *pa, const void *pb) {
    const Entry *x = *(Entry *const *)pa;
    const Entry *y = *(Entry *const *)pb;
    if (x->count != y->count) {
        return x->count > y->count ? -1 : 1; /* -count ascending */
    }
    int m = x->len < y->len ? x->len : y->len;
    int c = memcmp(x->w, y->w, (size_t)m);
    if (c != 0) {
        return c;
    }
    return x->len - y->len; /* a proper prefix sorts first */
}

static double run(long long n) {
    long long x = 42;
    char buf[32];
    for (long long i = 0; i < n; i++) {
        x = (x * 1103515245LL + 12345LL) % 2147483648LL;
        int len = make_word(x, buf);
        count_word(buf, len);
    }
    qsort(uniq, nuniq, sizeof(Entry *), cmp_entry);
    long long acc = 0;
    size_t top = nuniq < 10 ? nuniq : 10;
    for (size_t i = 0; i < top; i++) {
        acc += (long long)(i + 1) * uniq[i]->count + uniq[i]->len;
    }
    return (double)(acc * 100000LL + (long long)nuniq);
}

int main(int argc, char **argv) {
    long long n = argc > 1 ? strtoll(argv[1], NULL, 10) : DEFAULT_N;
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    double r = run(n);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double secs = (double)(t1.tv_sec - t0.tv_sec) + (double)(t1.tv_nsec - t0.tv_nsec) * 1e-9;
    printf("{\"kernel\": \"dictwords\", \"n\": %lld, \"checksum\": %.17g, \"seconds\": %.9f}\n", n, r, secs);
    return 0;
}
