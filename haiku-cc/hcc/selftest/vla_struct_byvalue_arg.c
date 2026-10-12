// EXPECT-EXIT: 0
// A struct with a VLA tail passed by value: the callee gets a copy.
int g(int n) {
    struct S { char t[n + 4]; };
    int f(struct S a) {
        a.t[0] = 'x';
        return a.t[0] == 'x';
    }
    struct S s1, s2;
    s1.t[0] = 'a';
    s2.t[0] = 'b';
    if (f(s1) != 1) return 1;
    if (s1.t[0] != 'a') return 2;
    if (s2.t[0] != 'b') return 3;
    return 0;
}
int main(void) { return g(3); }
