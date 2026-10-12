// EXPECT-EXIT: 0
// gcc evaluates the rhs call of `lhs op= f()` before reading lhs (value and address).
int g = 5;
int a[4];
unsigned int x[1] = { 2 };
int bar(void) { g = 100; return 1; }
unsigned int foo(void) { x[0] |= 128; return 1; }
int idx(void) { return 0; }
int one(void) { a[0] = 9; return 1; }
int main(void) {
    g += bar();
    if (g != 101) return 1;
    x[0] |= foo();
    if (x[0] != 131) return 2;
    a[idx()] += one();          
    if (a[0] != 10) return 3;   /* a[0] == 9 after one(), then += 1 */
    return 0;
}
