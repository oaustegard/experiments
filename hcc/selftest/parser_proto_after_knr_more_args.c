// EXPECT-ERROR: declares more arguments than previous old-style definition
// A prototype with more parameters than an earlier old-style `()` definition is an error.
void g() {}
void g(int);
int main(void) { return 0; }
