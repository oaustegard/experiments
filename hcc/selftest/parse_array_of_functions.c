// EXPECT-ERROR: [parse] declaration as array of functions
// An array element type must not be a function type (C11 6.7.6.2p1).
int x[3]();
int main(void) { return 0; }
