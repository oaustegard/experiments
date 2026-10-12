// EXPECT-ERROR: array size missing
// EXPECT-FLAGS: -std=c23
// C23: a file-scope static object with an incomplete array type and no initializer is an error.
static int sv[];
int main (void) { return 0; }
