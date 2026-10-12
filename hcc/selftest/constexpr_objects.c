// EXPECT-FLAGS: -std=c2x
// EXPECT-EXIT: 0
// C23 constexpr objects are constant expressions: block-scope scalars, structs
// (member access, copies) and null pointers can initialize static storage and
// integer constant expressions.
struct s { int a; int b[2]; };
constexpr struct s gs = { 3, { 4, 9 } };
constexpr void *gnull = 0;
int *gp = gnull;
int check(void) {
    constexpr int x = 7;
    constexpr struct s sv = { 3, { 4, 9 } };
    auto constexpr y = sv;
    constexpr int *np = (int *) 0;
    static int st = x + sv.a;
    enum { N = x + gs.a + 1 }; int arr[N] = { 0 };
    static_assert(y.a == 3);
    if (gp != 0 || np != 0) return 1;
    if (st != 10) return 2;
    if (sizeof arr != 11 * sizeof(int)) return 3;
    if ((constexpr struct s){1, {2, 3}}.b[1] != 3) return 4;
    return 0;
}
int main(void) { return check(); }
