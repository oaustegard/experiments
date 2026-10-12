// EXPECT-EXIT: 0
// The GNU spelling __complex__ must be accepted as a type specifier.
__complex__ double f (void)
{
  int a[4];
  __complex__ double c;
  a[1] = 0;
  c = a[1];
  return c;
}
int main (void)
{
  __complex__ double c;
  if (c = f ())
    return 1;
  return 0;
}
