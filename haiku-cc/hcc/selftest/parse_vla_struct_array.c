// EXPECT-EXIT: 0
/* Array of structs with a VLA member: storage is run-time sized and
   element strides (pointer arithmetic, indexing) use the bound struct size. */
int f(int n) {
    struct s { int x[n]; int y; } a[3];
    struct s *p = a;
    if (sizeof(a) != 3 * (4 * n + 4)) return 1;
    a[2].y = 9;
    p = p + 2;
    if (p->y != 9) return 2;
    if ((&a[2] - &a[0]) != 2) return 3;
    return 0;
}
int main(void) {
    return f(3) + f(5);
}
