// EXPECT-EXIT: 0
// __builtin_offsetof accepts GCC's `arr->member` designator (element 0 of arr).
struct S { int c; struct { float f; } sa[2]; };
char a[__builtin_offsetof(struct S, sa->f) == __builtin_offsetof(struct S, sa[0].f) ? 1 : -1];
int main(void) { return sizeof a == 1 ? 0 : 1; }
