// EXPECT-ERROR: [parse]
// Generic __atomic_exchange must not take a pointer to a function as its value operand (gcc PR 71514).
void foo(void) {}
int a, b;
void f(void) {
    __atomic_exchange(&a, &foo, &b, 0);
}
