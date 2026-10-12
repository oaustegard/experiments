// EXPECT-EXIT: 0
// A UCN inside a pp-number continues the number, so `.p` is not a macro use.
#define a(x) b(x)
#define b(x) 0
#define p )
int c = a(0À.p);
int main(void) { return c; }
