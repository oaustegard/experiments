// EXPECT-EXIT: 7
// Members without const, and whole structs without const members, stay assignable;
// only the enclosing struct that holds a const member is read-only.
struct in { const int k; };
struct out { int a; struct in i; };
struct plain { int x; int y; };
int main(void) {
    struct out o;
    struct plain p = { 1, 2 }, q = { 3, 4 };
    o.a = 5;
    p = q;
    p.x = p.x + o.a - 1;
    return p.x;
}
