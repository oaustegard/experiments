// EXPECT-ERROR: register name not specified
// File-scope register variable without an asm register name is invalid.
register int r0;
int main(void) { return 0; }
