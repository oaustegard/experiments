// EXPECT-EXIT: 0
// A file-scope prototype that differs from the builtin strcmp only in pointee
// qualifiers is accepted (gcc merely warns about it).
int strcmp(char *p, char *q);
int main(void) { return 0; }
