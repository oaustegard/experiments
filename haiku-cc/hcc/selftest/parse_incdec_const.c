// EXPECT-ERROR: [parse]
// ++/-- on a const-qualified lvalue must be rejected (gcc: increment of read-only variable).
const int b;
void f(void) { b++; }
