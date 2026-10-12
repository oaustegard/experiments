// EXPECT-ERROR: bit-field 'p' has invalid type
// A bit-field must have an integer, _Bool or enum type: pointers, floats and
// vectors (vector_size written after the width) are rejected.
struct S { int *p : 2; };
int main(void) { return 0; }
