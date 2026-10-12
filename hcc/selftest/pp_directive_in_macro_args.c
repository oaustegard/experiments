// EXPECT-EXIT: 0
// Directives inside a macro argument list are processed (C11 6.10.3p11).
#define f(x) x
int main(void) {
  int a = f (
#if f(1)
	0)
#else
	1)
#endif
  ;
  int b = f(2
#undef f
#define f - 1
  f);
  return !(a == 0 && b == 1);
}
