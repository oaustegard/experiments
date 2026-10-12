// EXPECT-EXIT: 0
// EXPECT-FLAGS: -std=c23
// C23 constexpr objects: the object itself is const (pointer qualifier, not pointee),
// and their values are constant expressions (members, auto locals, compound literals).
struct s { int a; int *p; double d; };
constexpr void *vp = 0;
constexpr int *vip = nullptr;
int *use_vp = vp;
int *use_vip = vip;
extern int *const *pcpi;
int *const *pcpi = &use_vip;
static_assert (sizeof (vip) == sizeof (int *));
int main (void) {
    constexpr struct s cs = { 7, 0, 2.5 };
    constexpr struct s cs2 = cs;
    constexpr auto n = 4;
    static int st[n];
    static_assert (cs.a == 7);
    static_assert (cs2.a == 7);
    static_assert ((constexpr struct s) { 9 }.a == 9);
    static_assert (n == 4);
    int *q = (int *) 0;
    return (q != 0) + (sizeof st / sizeof st[0] == 4 ? 0 : 1) + (use_vp != 0) + cs2.a - 7;
}
