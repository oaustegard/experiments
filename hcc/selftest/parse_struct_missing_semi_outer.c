// EXPECT-EXIT: 5
// GNU: a struct member list may omit the final ';' before '}', and a
// struct/union specifier may end the list without its ';' (anonymous member).
struct outer { struct { int a; } };
struct last { int x; int y };
int main(void) {
    struct outer o; struct last l;
    o.a = 3; l.x = 2; l.y = 0;
    return o.a + l.x + l.y;
}
