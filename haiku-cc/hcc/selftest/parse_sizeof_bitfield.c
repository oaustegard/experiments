// EXPECT-ERROR: [parse]
struct { int a : 1; } x;
int r = sizeof (x.a);
