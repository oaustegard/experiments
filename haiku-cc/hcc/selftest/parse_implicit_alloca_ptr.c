// EXPECT-EXIT: 0
// Undeclared alloca() returns void* (gcc builtin), not int.
int f(int n) { int *q = (int *)alloca(sizeof(n)); *q = n; return *q; }
int main(void) {
    return f(3) != 3;
}
