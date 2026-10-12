/* PR tree-optimization/81549 */
/* { dg-do run } */
/* { dg-options "-O2" } */

int a[4];

__attribute__((noipa)) int
f (int n)
{
  int t0 = a[0];
  int t1 = a[1];

  for (int i = 0; i < n; ++i)
    {
      t0 = t1;
      t1 = 2;
    }

  return t0;
}

__attribute__((noipa)) int
g (void)
{
  int t0 = a[0];
  int t1 = a[1];

  for (int i = 0; i < 1; ++i)
    {
      t0 = t1;
      t1 = 2;
    }

  return t0;
}

int
main (void)
{
  a[0] = 11;
  a[1] = 22;

  if (f (0) != 11 || f (1) != 22 || f (2) != 2 || g () != 22)
    __builtin_abort ();
  return 0;
}
