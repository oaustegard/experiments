/* PR tree-optimization/125234 */
/* { dg-do run } */

__attribute__((noinline))
int f0(int a, int b)
{
  return (a != b) & ((a | b) == 0);
}

__attribute__((noinline))
int f1(int a, int b)
{
  return (a == b) | ((a | b) != 0);
}


__attribute__((noinline))
int f0_(int a, int b)
{
  if (a != b)
    if ((a | b) == 0)
      return 1;
  return 0;
}

__attribute__((noinline))
int f1_(int a, int b)
{
  if (a == b)
    return 1;
  if ((a | b) != 0)
    return 1;
  return 0;
}


int
main()
{
  if (f0(0, 0))
    __builtin_abort ();
  if (!f1(0, 0))
    __builtin_abort ();
  if (f0_(0, 0))
    __builtin_abort ();
  if (!f1_(0, 0))
    __builtin_abort ();
}

