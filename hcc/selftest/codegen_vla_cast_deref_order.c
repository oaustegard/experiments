// EXPECT-EXIT: 0
// The VLA length binding of a pointer-to-VLA cast (a Comma) must run before the
// pointer arithmetic that reads it: `(*(int (*)[n])p)[1]` strides by n ints.
// Also: __auto_type of an array-typed (VLA) lvalue decays to a pointer.
int a[18];
int main(void) {
    int *p = a;
    int n = 2;
    a[2] = 7;
    if ((*(int (*)[9][n])p)[1][0] != 7) return 1;
    __auto_type xx = (*(int (*)[9][n])p);
    if (xx[1][0] != 7) return 2;
    return 0;
}
