// EXPECT-ERROR: [parse]
// Alignment above the x86-64 object file maximum (2**28) is an error.
int foo __attribute__ ((aligned(1 << 29))) = 20;
