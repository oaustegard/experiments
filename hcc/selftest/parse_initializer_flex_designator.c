// EXPECT-EXIT: 0
// Flexible array member in brace elision, nested designator continuation,
// and whole-value re-initialization by a later designator.
typedef struct { char a, b[]; } F;
F g = {1, 2, 3};
int x[2][3] = {1, 2, 3, 4, 5, 6, [0][1] = 7, 8, [0] = 9, [0] = 10, 11, [1][0] = 12};
struct P { int a, b; };
struct P y[] = {{1, 2}, [0].b = 3};
// whole-value re-init: y2[0] = s is then overridden by .b, so a becomes 0
struct P s0 = {1, 2};
struct P y2[] = {s0, [0].b = 3};
int main(void) {
    int bad = 0;
    bad += g.a != 1 || g.b[0] != 2 || g.b[1] != 3;
    bad += x[0][0] != 10 || x[0][1] != 11 || x[0][2] != 8;
    bad += x[1][0] != 12 || x[1][1] != 5;
    bad += y[0].a != 1 || y[0].b != 3;
    bad += y2[0].a != 0 || y2[0].b != 3;
    return bad != 0;
}
