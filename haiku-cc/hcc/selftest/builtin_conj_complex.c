// EXPECT-EXIT: 0
// __builtin_conj{f,,l} negates the imaginary part and keeps the real part.
extern void abort(void);
__complex__ float foo(void) {
  __complex__ float f = 1 + 1i;
  return __builtin_conjf(f);
}
int main(void) {
  __complex__ double d = __builtin_conj(2.0 + 3.0i);
  if (__real__ d != 2.0 || __imag__ d != -3.0)
    abort();
  if (__imag__ foo() != -1.0f)
    abort();
  return 0;
}
