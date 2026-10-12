
/* { dg-do compile } */
/* { dg-options "-O1 -fdump-tree-forwprop1-folding-raw" } */

static inline int
min (int a, int b)
{
  return a < b ? a : b;
}

static inline int
max (int a, int b)
{
  return a > b ? a : b;
}

int
f1 (int x, int y, int z)
{
  return min (z + y, x + y);
}

int
f2 (int x, int y, int z)
{
  return min (z - y, x - y);
}

int
f3 (int x, int y, int z)
{
  return max (z + y, x + y);
}

int
f4 (int x, int y, int z)
{
  return max (z - y, x - y);
}

/* { dg-final { scan-tree-dump-times "Applying pattern" 4 "forwprop1" } } */
/* { dg-final { scan-tree-dump-times "plus_expr" 2 "forwprop1" } } */
/* { dg-final { scan-tree-dump-times "minus_expr" 2 "forwprop1" } } */
