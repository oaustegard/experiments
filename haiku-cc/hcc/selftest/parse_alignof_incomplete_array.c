// EXPECT-ERROR: invalid application of '_Alignof' to incomplete type
// _Alignof on an incomplete array type (T[] with no bound) is a constraint violation.
int a = _Alignof(int[]);
