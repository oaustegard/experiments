// EXPECT-EXIT: 0
// A struct tag declared in a definition's parameter list is visible in the body.
int f(int n, struct S { char a[n]; } *p) {
    return (int)sizeof(struct S) - (int)sizeof(struct S) + n * 0 + (p == 0) - 1;
}
int main(void) { return f(3, 0); }
