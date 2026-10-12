// EXPECT-EXIT: 0
// An abstract declarator `(...)` nested in a parameter list is a variadic
// parameter list, not a grouped declarator.
int g(int (...));
int h(...) { return 0; }
int main(void) {
    return h() + h(1, 2);
}
