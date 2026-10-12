// EXPECT-EXIT: 0
// Identical string literals in one translation unit share storage.
char *p;
char *foo(void) { return p + 1; }
int main(void) {
    p = "Hello";
    if (foo() != "Hello" + 1) return 1;
    return 0;
}
