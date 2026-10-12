// EXPECT-ERROR: read-only location
// Assignment through a pointer to const, and to a member of a const struct, is an error.
struct S { int a; };
int main(void) {
    const int *p = 0;
    struct S s = { 1 };
    const struct S *cs = &s;
    p[0] = 1;
    (void)cs;
    return 0;
}
