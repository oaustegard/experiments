// EXPECT-EXIT: 0
// __builtin_va_arg_pack() inside an always_inline variadic wrapper takes the extra
// arguments of the call site, forwarded through a second wrapper.
static int sum4(int a, int b, int c, int d)
{
  return a * 1000 + b * 100 + c * 10 + d;
}
static inline __attribute__((__always_inline__)) int
inner(int a, ...)
{
  if (__builtin_va_arg_pack_len() != 3)
    return -1;
  return sum4(a, __builtin_va_arg_pack());
}
static inline __attribute__((__always_inline__)) int
outer(int a, ...)
{
  return inner(a, 7, __builtin_va_arg_pack());
}
int main(void)
{
  return outer(1, 2, 3) == 1723 ? 0 : 1;
}
