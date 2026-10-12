// EXPECT-ERROR: [parse]
// A block-scope `auto` function declaration without a definition is an error
// even when it is never called (GCC: "nested function declared but never defined").
void f(void) { auto int fn2(int); }
int main(void) { return 0; }
