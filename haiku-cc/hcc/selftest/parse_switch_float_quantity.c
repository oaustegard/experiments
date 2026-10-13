// EXPECT-ERROR: [parse]
// A switch controlling expression must have integer type (not float).
void f(void) {
    float x;
    switch (x) { case 0: break; }
}
