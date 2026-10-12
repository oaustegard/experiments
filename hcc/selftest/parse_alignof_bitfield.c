// EXPECT-ERROR: applied to a bit-field
// _Alignof on a bit-field member is a constraint violation.
struct { int a : 1; } x;
int r = _Alignof(x.a);
