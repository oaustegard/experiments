// EXPECT-EXIT: 0
// ?: of `int (*)[]` and `int (*)[2]` has the complete array type, so sizeof works.
int main(void) {
    int (*pi)[];
    int (*pf)[2];
    int i = 1;
    static int x[sizeof(*(i ? pi : pf))];
    return (int)sizeof(x) == 32 ? 0 : 1;
}
