// EXPECT-EXIT: 0
typedef unsigned short V __attribute__((vector_size(8)));
typedef V VI;
V a, b, c, d;
int main(void) {
    VI mask = { 3, 0, 5, 2 };
    a = (V){ 1, 2, 3, 4 };
    b = (V){ 10, 20, 30, 40 };
    c = __builtin_shuffle(a, mask);       /* one operand: index & 3 */
    d = __builtin_shuffle(a, b, mask);    /* two operands: bit 2 picks b */
    if (c[0] != 4 || c[1] != 1 || c[2] != 2 || c[3] != 3) return 1;
    if (d[0] != 4 || d[1] != 1 || d[2] != 20 || d[3] != 3) return 2;
    return 0;
}
