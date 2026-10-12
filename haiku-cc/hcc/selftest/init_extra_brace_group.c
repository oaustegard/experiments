// EXPECT-ERROR: [parse]
// An aggregate with no element left may not take a trailing brace group.
struct a { int x, y; };
struct b { struct a arr[1]; };
static struct b v = { {1, 2}, {3, 4} };
