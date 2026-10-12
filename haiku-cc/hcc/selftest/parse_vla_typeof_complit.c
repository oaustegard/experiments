// EXPECT-EXIT: 21
// The side effects of a VLA typeof operand inside a compound literal's type
// run once, before the literal (not duplicated by the parser's backtrack).
int a[1];
int main(void) {
    int i = 0;
    (typeof (++i, (int (*)[i])a)){&a} = 0;   /* i: 1 */
    (typeof (++i, (int (*)[i])a)){&a} += 1;  /* i: 2 */
    int n = 0;
    int (*p)[i] = (typeof (++n, (int (*)[i])a))a; /* n: 1 */
    return i * 10 + n + (p == (int (*)[i])a ? 0 : 9);
}
