// EXPECT-ERROR: conflicting types for 'y'
extern struct bar { float x; } y;
void f(void) {
    extern struct bar { int x; } y;
}
