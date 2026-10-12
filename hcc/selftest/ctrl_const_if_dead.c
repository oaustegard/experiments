// EXPECT-EXIT: 0
extern void link_error(void);
static int ok = 0;
static void bar(void) { ok = 1; }
static void foo(int x) {
    switch (x) {
    case 0:
        if (0) {
            link_error();
    case 1:
            bar();
        }
    }
}
static void baz(unsigned d) {
    int i;
    for (i = 0; i < 5; i += d)
        if (__builtin_constant_p(d))
            link_error();
}
int main(void) {
    foo(1);
    baz(1);
    if (!ok) return 1;
    return 0;
}
