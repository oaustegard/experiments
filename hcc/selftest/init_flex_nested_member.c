// EXPECT-ERROR: [parse]
// A trailing `T x[]` member may not be initialized inside a nested aggregate.
struct f { int w; int x[]; };
struct g { struct f f; };
struct g g2 = { { 0, { 1 } } };
