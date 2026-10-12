// EXPECT-FLAGS: -std=c2x
// EXPECT-ERROR: [sema] initializer element is not constant
int f(int a){ constexpr int c = a; return c; }
