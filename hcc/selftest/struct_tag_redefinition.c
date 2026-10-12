// EXPECT-ERROR: [parse]
// A complete struct tag may not be defined again in the same scope.
extern struct bar { int x; } b;
struct bar { int x; };
