// EXPECT-EXIT: 0
// __auto_type initialised from a VLA lvalue: the variable is a pointer to the
// element VLA (array-to-pointer decay), so its sizeof is pointer-sized.
int main(void) {
    int n = 3;
    int v[n][n];
    __auto_type r = &v[0];
    __auto_type q = v;
    if (sizeof(q) != sizeof(void *)) return 1;
    q[1][2] = 5;
    if (v[1][2] != 5) return 2;
    return 0;
}
