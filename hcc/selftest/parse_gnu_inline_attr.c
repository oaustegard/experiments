// EXPECT-EXIT: 1
// __attribute__((gnu_inline)) makes `extern inline` an inline-only definition
// (no external symbol), so a later plain `inline` definition is not a redefinition.
__attribute__((gnu_inline)) extern inline int f(void) { return 0; }
__attribute__((gnu_inline)) inline int f(void) { return 1; }
int main(void) { return f(); }
