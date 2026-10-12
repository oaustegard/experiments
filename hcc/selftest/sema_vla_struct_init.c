// EXPECT-ERROR: [parse]
void f(int c) {
    struct s { int x[c]; int y; } v = { 1, 2 };
}
