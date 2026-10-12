// EXPECT-EXIT: 0
// Bit-fields after a run-time-sized member and arrays of such structs (stride and storage).
struct S { char d[5]; int a : 3; int b : 17; int c : 12; };
void f(int n) {
    struct { char d[n]; int a : 3; int b : 17; int c : 12; } v;
    v.b = -5; if (v.b != -5) __builtin_trap();
    struct S A[2][3];
    struct S *p = &A[0][0];
    int i, j;
    for (i = 0; i < 2; i++)
        for (j = 0; j < 3; j++)
            A[i][j].b = 2;
    A[1][2].c = -7;
    if (A[1][2].c != -7 || A[0][0].b != 2 || A[1][1].b != 2) __builtin_trap();
    if ((char *)&A[0][1] - (char *)&A[0][0] != (long)sizeof(struct S)) __builtin_trap();
    (void)p; (void)n;
}
int main(void) {
    f(0);
    return 0;
}
