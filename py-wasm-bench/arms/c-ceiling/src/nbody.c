/* N-body, hand-translated from kernels/nbody.py. Same operation order. */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define PI 3.141592653589793
#define SOLAR_MASS (4.0 * PI * PI)
#define DAYS_PER_YEAR 365.24
#define NB 5
#define DEFAULT_N 200000

static double bodies[NB][7]; /* x y z vx vy vz mass */

static void make_bodies(void) {
    bodies[0][0] = 0.0; bodies[0][1] = 0.0; bodies[0][2] = 0.0;
    bodies[0][3] = 0.0; bodies[0][4] = 0.0; bodies[0][5] = 0.0;
    bodies[0][6] = SOLAR_MASS;

    bodies[1][0] = 4.84143144246472090e+00; bodies[1][1] = -1.16032004402742839e+00;
    bodies[1][2] = -1.03622044471123109e-01;
    bodies[1][3] = 1.66007664274403694e-03 * DAYS_PER_YEAR;
    bodies[1][4] = 7.69901118419740425e-03 * DAYS_PER_YEAR;
    bodies[1][5] = -6.90460016972063023e-05 * DAYS_PER_YEAR;
    bodies[1][6] = 9.54791938424326609e-04 * SOLAR_MASS;

    bodies[2][0] = 8.34336671824457987e+00; bodies[2][1] = 4.12479856412430479e+00;
    bodies[2][2] = -4.03523417114321381e-01;
    bodies[2][3] = -2.76742510726862411e-03 * DAYS_PER_YEAR;
    bodies[2][4] = 4.99852801234917238e-03 * DAYS_PER_YEAR;
    bodies[2][5] = 2.30417297573763929e-05 * DAYS_PER_YEAR;
    bodies[2][6] = 2.85885980666130812e-04 * SOLAR_MASS;

    bodies[3][0] = 1.28943695621391310e+01; bodies[3][1] = -1.51111514016986312e+01;
    bodies[3][2] = -2.23307578892655734e-01;
    bodies[3][3] = 2.96460137564761618e-03 * DAYS_PER_YEAR;
    bodies[3][4] = 2.37847173959480950e-03 * DAYS_PER_YEAR;
    bodies[3][5] = -2.96589568540237556e-05 * DAYS_PER_YEAR;
    bodies[3][6] = 4.36624404335156298e-05 * SOLAR_MASS;

    bodies[4][0] = 1.53796971148509165e+01; bodies[4][1] = -2.59193146099879641e+01;
    bodies[4][2] = 1.79258772950371181e-01;
    bodies[4][3] = 2.68067772490389322e-03 * DAYS_PER_YEAR;
    bodies[4][4] = 1.62824170038242295e-03 * DAYS_PER_YEAR;
    bodies[4][5] = -9.51592254519715870e-05 * DAYS_PER_YEAR;
    bodies[4][6] = 5.15138902046611451e-05 * SOLAR_MASS;
}

static void offset_momentum(void) {
    double px = 0.0, py = 0.0, pz = 0.0;
    for (int i = 0; i < NB; i++) {
        px += bodies[i][3] * bodies[i][6];
        py += bodies[i][4] * bodies[i][6];
        pz += bodies[i][5] * bodies[i][6];
    }
    bodies[0][3] = -px / SOLAR_MASS;
    bodies[0][4] = -py / SOLAR_MASS;
    bodies[0][5] = -pz / SOLAR_MASS;
}

static double energy(void) {
    double e = 0.0;
    for (int i = 0; i < NB; i++) {
        double *b = bodies[i];
        e += 0.5 * b[6] * (b[3] * b[3] + b[4] * b[4] + b[5] * b[5]);
        for (int j = i + 1; j < NB; j++) {
            double *c = bodies[j];
            double dx = b[0] - c[0];
            double dy = b[1] - c[1];
            double dz = b[2] - c[2];
            e -= (b[6] * c[6]) / pow(dx * dx + dy * dy + dz * dz, 0.5);
        }
    }
    return e;
}

static void advance(double dt, long long steps) {
    for (long long s = 0; s < steps; s++) {
        for (int i = 0; i < NB; i++) {
            double *b = bodies[i];
            for (int j = i + 1; j < NB; j++) {
                double *c = bodies[j];
                double dx = b[0] - c[0];
                double dy = b[1] - c[1];
                double dz = b[2] - c[2];
                double d2 = dx * dx + dy * dy + dz * dz;
                double mag = dt / (d2 * pow(d2, 0.5));
                double bm = b[6] * mag;
                double cm = c[6] * mag;
                b[3] -= dx * cm;
                b[4] -= dy * cm;
                b[5] -= dz * cm;
                c[3] += dx * bm;
                c[4] += dy * bm;
                c[5] += dz * bm;
            }
        }
        for (int i = 0; i < NB; i++) {
            double *b = bodies[i];
            b[0] += dt * b[3];
            b[1] += dt * b[4];
            b[2] += dt * b[5];
        }
    }
}

static double run(long long n) {
    make_bodies();
    offset_momentum();
    advance(0.01, n);
    return energy();
}

int main(int argc, char **argv) {
    long long n = argc > 1 ? strtoll(argv[1], NULL, 10) : DEFAULT_N;
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    double r = run(n);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double secs = (double)(t1.tv_sec - t0.tv_sec) + (double)(t1.tv_nsec - t0.tv_nsec) * 1e-9;
    printf("{\"kernel\": \"nbody\", \"n\": %lld, \"checksum\": %.17g, \"seconds\": %.9f}\n", n, r, secs);
    return 0;
}
