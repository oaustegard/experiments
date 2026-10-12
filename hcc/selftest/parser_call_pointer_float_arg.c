// EXPECT-ERROR: [parse]
// A pointer is never converted to a floating parameter (C11 6.5.16.1p1).
float f(float x);
float g(const char *p) { return f(p); }
int main(void) { return 0; }
