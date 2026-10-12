// EXPECT-ERROR: [parse]
int main(void) {
    int foo ();
    const int foo () { return 0; }
    return foo();
}
