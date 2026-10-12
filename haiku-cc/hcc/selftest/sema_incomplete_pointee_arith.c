// EXPECT-ERROR: arithmetic on pointer to an incomplete type
// Pointer arithmetic on a pointer to an incomplete struct must be rejected (C11 6.5.6p1).
struct a;
struct a *q;
void f(void) { q++; }
