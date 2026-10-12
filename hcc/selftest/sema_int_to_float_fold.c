// EXPECT-EXIT: 0
// Folded int -> float conversions round once from the exact integer:
// 16777217 is not representable in float (-> 16777216), and the
// unsigned long long maximum is a large positive double, not -1.
extern void abort(void);
int main(void) {
  if (16777217L != (float)16777217e0)
    abort();
  if ((double)18446744073709551615ULL < 1.84467440737095e+19 ||
      (double)18446744073709551615ULL > 1.84467440737096e+19)
    abort();
  return 0;
}
