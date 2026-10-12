// EXPECT-ERROR: read-only variable used as 'asm' output
const int i = 0;

void f(void)
{
  __asm__ __volatile__ ("" : "=m" (i));
}
