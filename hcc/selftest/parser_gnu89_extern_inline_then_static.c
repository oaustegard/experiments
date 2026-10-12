// EXPECT-EXIT: 0
// gnu89 `extern inline` bodies provide no external symbol, so a file-scope `static`
// declaration (with or without a body) may follow them (GCC -fgnu89-inline).
__attribute__((gnu_inline)) extern inline int foo(void) { return 0; }
__attribute__((gnu_inline)) extern inline int bar(void) { return 0; }
static int bar(void) { return foo() + 1; }
int main(void) { return bar() - 1; }
