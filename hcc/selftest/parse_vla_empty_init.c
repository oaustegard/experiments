// EXPECT-EXIT: 0
// Empty braces on a variable-length array zero-fill it (GNU/C23 form).
int f(int n) {
    int v[n] = {};
    int ok = 1;
    for (int i = 0; i < n; i++) ok = ok && v[i] == 0;
    return ok;
}
int main(void) {
    return f(5) ? 0 : 1;
}
