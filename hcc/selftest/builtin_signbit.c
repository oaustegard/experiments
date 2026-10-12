// EXPECT-EXIT: 0
int main(void) {
  double pz = 0.0, nz = -0.0, n = -2.5;
  float f = -1.0f;
  long double ld = -3.0L;
  if (__builtin_signbit(pz)) return 1;
  if (!__builtin_signbit(nz)) return 2;
  if (!__builtin_signbit(n)) return 3;
  if (!__builtin_signbit(f)) return 4;
  if (!__builtin_signbit(ld)) return 5;
  if (__builtin_signbit(1.0)) return 6;
  return 0;
}
