// EXPECT-EXIT: 0
// EXPECT-FLAGS: -std=c23
// C23: `void f();` is the same as `void f(void);`, so _Generic selects
// the (void) association, not the (int) one.
void f();
_Static_assert(_Generic(f, void (*)(int): 1, void (*)(void): 2, default: 3) == 2, "");
int main(void) { return 0; }
