/* Mandelbrot escape counts, hand-translated from kernels/mandelbrot.py. */
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define DEFAULT_N 500

static double run(long long n) {
    const long long max_iter = 50;
    long long inside = 0;
    long long total_iters = 0;
    for (long long py = 0; py < n; py++) {
        double ci = 2.0 * (double)py / (double)n - 1.0;
        for (long long px = 0; px < n; px++) {
            double cr = 2.0 * (double)px / (double)n - 1.5;
            double zr = 0.0;
            double zi = 0.0;
            long long k = 0;
            while (k < max_iter) {
                double zr2 = zr * zr;
                double zi2 = zi * zi;
                if (zr2 + zi2 > 4.0) {
                    break;
                }
                zi = 2.0 * zr * zi + ci;
                zr = zr2 - zi2 + cr;
                k++;
            }
            total_iters += k;
            if (k == max_iter) {
                inside++;
            }
        }
    }
    return (double)(inside * 1000000LL + total_iters % 1000000LL);
}

int main(int argc, char **argv) {
    long long n = argc > 1 ? strtoll(argv[1], NULL, 10) : DEFAULT_N;
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    double r = run(n);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double secs = (double)(t1.tv_sec - t0.tv_sec) + (double)(t1.tv_nsec - t0.tv_nsec) * 1e-9;
    printf("{\"kernel\": \"mandelbrot\", \"n\": %lld, \"checksum\": %.17g, \"seconds\": %.9f}\n", n, r, secs);
    return 0;
}
