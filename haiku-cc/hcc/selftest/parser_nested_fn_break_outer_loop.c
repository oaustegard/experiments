// EXPECT-ERROR: [parse]
// break/continue inside a nested function must not bind to a loop or switch of the enclosing function.
void f(int a) {
    while (a) {
        void g(void) { break; }
    }
}
