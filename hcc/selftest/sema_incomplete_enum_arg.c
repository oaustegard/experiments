// EXPECT-ERROR: has an incomplete type
// An incomplete enum object passed as a call argument is rejected.
enum e;
extern enum e ve;
void g(int);
void f(void) { g(ve); }
