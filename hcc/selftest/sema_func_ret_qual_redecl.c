// EXPECT-EXIT: 4
// Top-level const/volatile on a function's return type does not change compatibility.
int f(void);
const int f(void) { return 4; }
volatile int g(void);
int g(void) { return 0; }
int main(void) { return f(); }
