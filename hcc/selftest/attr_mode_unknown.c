// EXPECT-ERROR: [parse] unknown machine mode
// __attribute__((mode(x))) with an identifier that is not a GCC machine mode is rejected.
int a __attribute__((__mode__(bogus_mode)));
