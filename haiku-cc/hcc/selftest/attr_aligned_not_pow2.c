// EXPECT-ERROR: [parse] requested alignment '3' is not a positive power of 2
// Alignment that is not a power of two is rejected (GCC: not a positive power of 2).
int foo __attribute__ ((aligned(3))) = 20;
