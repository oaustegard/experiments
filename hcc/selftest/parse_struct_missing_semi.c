// EXPECT-EXIT: 7
// GNU: a struct member list may omit the final ';' before '}', and a scalar
// initializer in braces may have excess elements.
struct inner { int v; };
struct outer { struct { int a; } }; 
struct last { int x; int y }; 
int main(void) {
    struct outer o; struct last l;
    o.a = 3; l.x = 2; l.y = 2;
    int *p = (int *){ 0, 1 };
    (void)p;
    return o.a + l.x + l.y;
}
