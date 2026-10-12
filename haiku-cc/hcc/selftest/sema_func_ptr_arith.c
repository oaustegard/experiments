// EXPECT-EXIT: 0
// A function designator used as an operand of + or - decays to a pointer.
static int f(void) { return 1; }
int main(void) {
    long n = 2;
    char *p = (char *)f + n;
    char *q = (char *)(f + 1);
    if (p - (char *)f != 2) return 1;
    if (q - (char *)f != 1) return 2;
    if (((char *)f + 3) - (char *)f != 3) return 3;
    return 0;
}
