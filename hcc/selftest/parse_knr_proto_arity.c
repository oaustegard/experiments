// EXPECT-ERROR: doesn't match prototype
// K&R definition with one parameter does not match the earlier two-parameter prototype.
int g (int, int);
int g (a) int a; { return a; }
