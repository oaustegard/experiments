// EXPECT-ERROR: [parse]
// EXPECT-ERROR: type is unavailable: Do not use
// A struct defined with __attribute__((unavailable("msg"))) is an unavailable type:
// naming an object of that type (here via typeof) is an error.
struct { int a; } __attribute__((unavailable ("Do not use"))) x;
typeof(x) y;
