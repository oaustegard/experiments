// EXPECT-EXIT: 0
// va_arg of __int128 consumes two GP slots (or the overflow area when fewer remain).
#include <stdarg.h>
static __int128 get (int n, ...)
{
  va_list ap;
  __int128 r;
  va_start (ap, n);
  while (n--)
    va_arg (ap, int);
  r = va_arg (ap, __int128);
  va_end (ap);
  return r;
}
int main (void)
{
  __int128 v = ((__int128) 0x1234 << 64) | 0x5678;
  if (get (0, v) != v) return 1;
  if (get (2, 0, 0, v) != v) return 2;
  if (get (5, 0, 0, 0, 0, 0, v) != v) return 3;
  return 0;
}
