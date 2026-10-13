// EXPECT-ERROR: [parse]
// EXPECT-ERROR: type is unavailable
// A declarator naming an unavailable struct type (even through a pointer) is an error;
// GCC rejects the declaration itself, not only later uses of the name.
struct { int a; } __attribute__((unavailable)) *p;
