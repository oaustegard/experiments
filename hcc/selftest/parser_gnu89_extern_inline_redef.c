// EXPECT-ERROR: [parse] redefinition of 'foo'
// Two gnu89-style `extern inline` bodies (gnu_inline) for one function are a redefinition.
__attribute__((gnu_inline)) extern inline int foo(void) { return 2; }
__attribute__((gnu_inline)) extern inline int foo(void) { return 1; }
int main(void) { return foo(); }
