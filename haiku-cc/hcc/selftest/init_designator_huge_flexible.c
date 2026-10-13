// EXPECT-ERROR: [parse]
// A huge designator index on an incomplete array must be rejected, not looped over.
static char *name[] = { [0x8000000000000000] = "bar" };
