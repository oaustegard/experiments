// EXPECT-ERROR: invalid use of undefined type
// Compound literal of an incomplete struct type is a constraint violation.
struct S {};
extern int foo (struct S);
int main () { foo ((struct T) {}); return 0; }
