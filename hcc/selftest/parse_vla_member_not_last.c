// EXPECT-EXIT: 0
/* GNU struct with variably modified members before other members: the
   offsets of the members after a run-time-sized one are run-time values. */
int f(int a) {
    struct { int p[a], i; } l;
    struct { char c; int q[a]; long z; char e; } m;
    struct { double x[a], y[a]; } d;
    l.i = 5;
    m.z = 1234567; m.e = 9; m.c = 1; m.q[a - 1] = 3;
    d.y[a - 1] = 2.5;
    if (sizeof(l) != 4 * a + 4) return 1;
    if (sizeof(m) != 40 && a == 5) return 2;
    if (sizeof(d) != 16 * a) return 3;
    if (l.i != 5) return 4;
    if ((char *)&l.i - (char *)&l != 4 * a) return 5;
    if ((char *)&m.z - (char *)&m != 8 * ((4 * a + 4 + 7) / 8)) return 6;
    if (m.z != 1234567 || m.e != 9 || m.q[a - 1] != 3) return 7;
    if (d.y[a - 1] != 2.5) return 8;
    return 0;
}
int main(void) {
    return f(3) + f(5);
}
