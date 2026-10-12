// EXPECT-EXIT: 0
struct P { int x; char c; long y; };
struct S { int a, b; };
struct S mk(int a, int b) { struct S s = {a, b}; return s; }
int sum(struct S s) { return s.a + s.b; }
union U { int i; char c[4]; };
int main(void) {
    struct P p = {1, 'a', 3};
    struct P *pp = &p;
    if (sizeof(struct P) != 16) return 1;
    if (pp->y != 3 || p.c != 'a') return 2;
    struct S s = mk(3, 4);
    if (sum(s) != 7) return 3;
    union U u; u.i = 0x01020304;
    if (u.c[0] != 4) return 4;
    struct P q = p; q.x = 9;
    if (p.x != 1 || q.x != 9) return 5;
    return 0;
}
