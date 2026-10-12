// EXPECT-ERROR: [parse]
// Block-scope redeclaration with an incompatible type (char vs int) is an error.
int f(void) {
    char c;
    int c = 1;
    return c;
}
int main(void) { return f(); }
