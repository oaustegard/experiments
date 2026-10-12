// EXPECT-EXIT: 0
/* GNU struct with a trailing variably-modified member: run-time size, member
   offset static, storage allocated like a VLA. */
int f(int n) {
    int m = 0;
    struct s { int y; int x[n]; } v;
    struct s *p = &v;
    v.x[n - 1] = 7;
    if (sizeof(v) != 4 + 4 * n) return 1;
    if (sizeof(struct s) != 4 + 4 * n) return 2;
    if (p->x[n - 1] != 7) return 3;
    if (sizeof(typeof(*({ m = 3; &v; }))) != 4 + 4 * n) return 4;
    if (m != 3) return 5;
    return 0;
}
int main(void) {
    return f(5) + f(1) * 0;
}
