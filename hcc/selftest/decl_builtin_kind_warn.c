// EXPECT-EXIT: 0
// A builtin redeclared with a different return kind only warns (-Wbuiltin-declaration-mismatch).
char *strcmp(const char *, const char *);
int main(void) { return 0; }
