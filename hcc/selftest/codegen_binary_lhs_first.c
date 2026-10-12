// EXPECT-EXIT: 0
// Binary operators evaluate their operands left to right (as gcc -O0 does),
// so side effects in both operands of `-` and `*`/`+` happen lhs first.
static int c;
static int nx(void) { return ++c; }
int main(void) {
    int a = nx() - nx();
    int b = nx() * 10 + nx();
    if (a != -1) return 1;
    if (b != 34) return 2;
    return 0;
}
