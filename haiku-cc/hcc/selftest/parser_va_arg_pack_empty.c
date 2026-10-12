// EXPECT-EXIT: 0
// __builtin_va_arg_pack() in an always_inline variadic wrapper called with no extra
// arguments: the clone has no pack parameters and __builtin_va_arg_pack_len() is 0.
static int zero(int a)
{
  return a;
}
static inline __attribute__((__always_inline__)) int
wrap(int a, ...)
{
  if (__builtin_va_arg_pack_len() != 0)
    return -1;
  return zero(a + 1) + zero(__builtin_va_arg_pack_len());
}
int main(void)
{
  return wrap(4) == 5 ? 0 : 1;
}
