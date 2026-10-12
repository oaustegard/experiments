// EXPECT-EXIT: 0
// Valid unary '&' operands: compound literals, members of lvalues, string literals.
struct s { int m; };
int main(void) {
    struct s v = { 1 };
    int *p = &(int){ 2 };
    int *q = &(struct s){ 3 }.m;
    int *r = &v.m;
    struct s *sp = &v;
    int *t = &sp->m;
    char *u = &"abc"[1];
    int a[2] = { 0, 0 };
    int *w = &a[1];
    int *x = &(v.m);
    return (*p == 2 && *q == 3 && r == &v.m && t == &v.m && *u == 'b' && w == &a[1] && x == &v.m) ? 0 : 1;
}
