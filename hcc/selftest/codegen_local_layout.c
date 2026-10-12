// EXPECT-EXIT: 0
// First-declared local must sit at the lowest address (as gcc -O0 does).
int main(void) {
    _Alignas(32) int *x, *y;
    long d = (char *)&y - (char *)&x;
    return d != 32;
}
