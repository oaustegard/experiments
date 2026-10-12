/* { dg-additional-options -O2 } */
/* { dg-additional-options -fdump-tree-optimized } */

int ll (signed a)
{
  int d = a >> 1;
  return d == 0;
}

int ll1 (signed a)
{
  int d = a & ~1;
  return d == 0;
}

int ll2 (signed a)
{
  unsigned aa = a;
  return aa <= 1;
}

int ll3 (signed a)
{
  int d = a >> 1;
  return d != 0;
}

int ll4 (signed a)
{
  int d = a & ~1;
  return d != 0;
}

int ll5 (signed a)
{
  unsigned aa = a;
  return aa > 1;
}

/* { dg-final { scan-tree-dump-times " >> " 0 optimized } } */
/* { dg-final { scan-tree-dump-times " <= " 3 optimized } } */
/* { dg-final { scan-tree-dump-times " > " 3 optimized } } */
