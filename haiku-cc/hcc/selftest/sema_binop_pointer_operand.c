// EXPECT-ERROR: [sema]
// A multiplicative/bitwise operator rejects a pointer operand (no silent conversion).
void f(void) {
  float t = 1;
  int const *s = 0;
  t / s;
}
