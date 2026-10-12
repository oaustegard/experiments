// EXPECT-ERROR: [parse]
// A negative array designator index is outside the array bounds.
char font[256][8] = { [-1] = { 0 } };
