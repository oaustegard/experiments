// EXPECT-EXIT: 7
// Pointers to incomplete structs, flexible array members and members of
// later-completed structs are valid.
struct later;
struct holder { struct later *p; int n; int tail[]; };
struct later { int v; };
struct outer { struct later l; struct later arr[2]; };
int main(void) {
    struct outer o;
    o.l.v = 3; o.arr[1].v = 4;
    return o.l.v + o.arr[1].v;
}
