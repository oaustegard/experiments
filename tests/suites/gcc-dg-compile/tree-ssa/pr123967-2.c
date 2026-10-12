/* { dg-additional-options -O2 } */
/* { dg-additional-options -fdump-tree-optimized } */

unsigned f1 (unsigned x, unsigned m)
{
  if (x)
    m |= 0x3;
  else
    m = m & ~0x3;

  return m + 1;
}

unsigned f2 (unsigned x, int m)
{
  int mask = 0x3;

  if (x)
    m |= mask;
  else
    m = m & ~mask;

  return m + 1;
}
/* { dg-final { scan-tree-dump-times " \\\| 3" 0 optimized } } */
