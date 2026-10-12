// EXPECT-EXIT: 0
// Extended asm: float/double operands in "r" constraints travel as raw bits;
// multi-alternative constraints ("=r,m") pick the register alternative.
extern void abort(void);
int main(void) {
  double x = 4, y;
  float f = 1.5f;
  asm volatile("" : "=r,r"(x), "=r,r"(y) : "%0,0"(x), "m,r"(8));
  asm("" : "+r"(f));
  if (x != 4.0 || f != 1.5f) abort();
  return 0;
}
