/* Dense n x n matmul, i-k-j order, hand-translated from kernels/matmul.py.
 * Matrices are flat row-major double arrays (the list-of-lists become n*n blocks). */
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define DEFAULT_N 200

static double *make(long long n, long long seed) {
    double *m = malloc((size_t)(n * n) * sizeof(double));
    for (long long i = 0; i < n; i++) {
        for (long long j = 0; j < n; j++) {
            m[i * n + j] = (double)((i * 31 + j * 17 + seed) % 97) / 97.0;
        }
    }
    return m;
}

static double *matmul(const double *a, const double *b, long long n) {
    double *c = malloc((size_t)(n * n) * sizeof(double));
    for (long long i = 0; i < n; i++) {
        double *ci = c + i * n;
        for (long long j = 0; j < n; j++) {
            ci[j] = 0.0;
        }
        const double *ai = a + i * n;
        for (long long k = 0; k < n; k++) {
            double aik = ai[k];
            const double *bk = b + k * n;
            for (long long j = 0; j < n; j++) {
                ci[j] += aik * bk[j];
            }
        }
    }
    return c;
}

static double run(long long n) {
    double *a = make(n, 1);
    double *b = make(n, 2);
    double *c = matmul(a, b, n);
    double s = 0.0;
    for (long long i = 0; i < n; i++) {
        for (long long j = 0; j < n; j++) {
            s += c[i * n + j] * (double)((i + j) % 5 + 1);
        }
    }
    free(a);
    free(b);
    free(c);
    return s;
}

int main(int argc, char **argv) {
    long long n = argc > 1 ? strtoll(argv[1], NULL, 10) : DEFAULT_N;
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    double r = run(n);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double secs = (double)(t1.tv_sec - t0.tv_sec) + (double)(t1.tv_nsec - t0.tv_nsec) * 1e-9;
    printf("{\"kernel\": \"matmul\", \"n\": %lld, \"checksum\": %.17g, \"seconds\": %.9f}\n", n, r, secs);
    return 0;
}
