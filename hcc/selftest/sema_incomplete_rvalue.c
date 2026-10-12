// EXPECT-ERROR: has an incomplete type
// An lvalue of incomplete struct type used as a value (expression statement) is rejected.
struct s;
extern struct s vs;
void f(void) { vs; }
