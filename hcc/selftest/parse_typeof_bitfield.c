// EXPECT-ERROR: [parse]
// typeof applied to a bit-field member is a constraint violation.
struct s { int i : 2; } x;
__typeof__(x.i) j;
int main(void) { return 0; }
