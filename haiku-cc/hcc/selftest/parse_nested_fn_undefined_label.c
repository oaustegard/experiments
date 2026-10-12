// EXPECT-ERROR: [parse]
// A goto inside a nested function to a label that exists in neither the
// nested function nor the enclosing function is still an error.
int main(void) {
    void f(void) {
        goto nowhere;
    }
    f();
    return 0;
}
