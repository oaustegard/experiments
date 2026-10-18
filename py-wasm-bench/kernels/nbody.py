"""N-body (benchmarks-game shape, 5 bodies). Float-heavy, list indexing."""
import json
import sys
import time

PI = 3.141592653589793
SOLAR_MASS = 4.0 * PI * PI
DAYS_PER_YEAR = 365.24


def make_bodies() -> list[list[float]]:
    # x, y, z, vx, vy, vz, mass
    return [
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, SOLAR_MASS],
        [4.84143144246472090e+00, -1.16032004402742839e+00, -1.03622044471123109e-01,
         1.66007664274403694e-03 * DAYS_PER_YEAR, 7.69901118419740425e-03 * DAYS_PER_YEAR,
         -6.90460016972063023e-05 * DAYS_PER_YEAR, 9.54791938424326609e-04 * SOLAR_MASS],
        [8.34336671824457987e+00, 4.12479856412430479e+00, -4.03523417114321381e-01,
         -2.76742510726862411e-03 * DAYS_PER_YEAR, 4.99852801234917238e-03 * DAYS_PER_YEAR,
         2.30417297573763929e-05 * DAYS_PER_YEAR, 2.85885980666130812e-04 * SOLAR_MASS],
        [1.28943695621391310e+01, -1.51111514016986312e+01, -2.23307578892655734e-01,
         2.96460137564761618e-03 * DAYS_PER_YEAR, 2.37847173959480950e-03 * DAYS_PER_YEAR,
         -2.96589568540237556e-05 * DAYS_PER_YEAR, 4.36624404335156298e-05 * SOLAR_MASS],
        [1.53796971148509165e+01, -2.59193146099879641e+01, 1.79258772950371181e-01,
         2.68067772490389322e-03 * DAYS_PER_YEAR, 1.62824170038242295e-03 * DAYS_PER_YEAR,
         -9.51592254519715870e-05 * DAYS_PER_YEAR, 5.15138902046611451e-05 * SOLAR_MASS],
    ]


def offset_momentum(bodies: list[list[float]]) -> None:
    px = 0.0
    py = 0.0
    pz = 0.0
    for b in bodies:
        px += b[3] * b[6]
        py += b[4] * b[6]
        pz += b[5] * b[6]
    sun = bodies[0]
    sun[3] = -px / SOLAR_MASS
    sun[4] = -py / SOLAR_MASS
    sun[5] = -pz / SOLAR_MASS


def energy(bodies: list[list[float]]) -> float:
    e = 0.0
    nb = len(bodies)
    for i in range(nb):
        b = bodies[i]
        e += 0.5 * b[6] * (b[3] * b[3] + b[4] * b[4] + b[5] * b[5])
        for j in range(i + 1, nb):
            c = bodies[j]
            dx = b[0] - c[0]
            dy = b[1] - c[1]
            dz = b[2] - c[2]
            e -= (b[6] * c[6]) / (dx * dx + dy * dy + dz * dz) ** 0.5
    return e


def advance(bodies: list[list[float]], dt: float, steps: int) -> None:
    nb = len(bodies)
    for _ in range(steps):
        for i in range(nb):
            b = bodies[i]
            for j in range(i + 1, nb):
                c = bodies[j]
                dx = b[0] - c[0]
                dy = b[1] - c[1]
                dz = b[2] - c[2]
                d2 = dx * dx + dy * dy + dz * dz
                mag = dt / (d2 * d2 ** 0.5)
                bm = b[6] * mag
                cm = c[6] * mag
                b[3] -= dx * cm
                b[4] -= dy * cm
                b[5] -= dz * cm
                c[3] += dx * bm
                c[4] += dy * bm
                c[5] += dz * bm
        for i in range(nb):
            b = bodies[i]
            b[0] += dt * b[3]
            b[1] += dt * b[4]
            b[2] += dt * b[5]


def run(n: int) -> float:
    bodies = make_bodies()
    offset_momentum(bodies)
    advance(bodies, 0.01, n)
    return energy(bodies)


DEFAULT_N = 200000

if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_N
    t0 = time.perf_counter()
    r = run(n)
    t1 = time.perf_counter()
    print(json.dumps({"kernel": "nbody", "n": n, "checksum": r, "seconds": t1 - t0}))
