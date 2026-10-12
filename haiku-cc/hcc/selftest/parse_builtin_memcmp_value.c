// EXPECT-EXIT: 0
// __builtin_memcmp named as a value (no call) resolves to the libc function.
int f (int flag)
{
  int (*p) (const void *, const void *, unsigned long);
  p = flag ? __builtin_memcmp : __builtin_memcmp;
  return p ("a", "a", 1);
}
int main (void) { return f (1); }
