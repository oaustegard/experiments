// EXPECT-ERROR: number of arguments doesn't match prototype
// Old-style definition with fewer parameters than the earlier prototype.
int g(int, int);
int g(a) int a; { return a; }
int main(void) { return 0; }
