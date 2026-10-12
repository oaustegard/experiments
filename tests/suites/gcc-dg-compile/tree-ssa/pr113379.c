/* { dg-do compile } */
/* { dg-options "-O1 -fdump-tree-optimized" } */

int
f1 (int a, int b)
{
  int min = a < b ? a : b;
  int max = a > b ? a : b;
  return min <= max;
}

int
f2 (int a, int b)
{
  int min = a < b ? a : b;
  int max = a > b ? a : b;
  return min > max;
}

int
f3 (int a, int b)
{
  int max = a > b ? a : b;
  int min = a < b ? a : b;
  return max >= min;
}

int
f4 (int a, int b)
{
  int max = a > b ? a : b;
  int min = a < b ? a : b;
  return max < min;
}

int
f5 (int a, int b)
{
  int min = a < b ? a : b;
  int max = a > b ? a : b;
  return min < max;
}

int
f6 (int a, int b)
{
  int min = a < b ? a : b;
  int max = a > b ? a : b;
  return min >= max;
}

int
f7 (int a, int b)
{
  int max = a > b ? a : b;
  int min = a < b ? a : b;
  return max > min;
}

int
f8 (int a, int b)
{
  int max = a > b ? a : b;
  int min = a < b ? a : b;
  return max <= min;
}

/* { dg-final { scan-tree-dump-not "MIN_EXPR" "optimized" } } */
/* { dg-final { scan-tree-dump-not "MAX_EXPR" "optimized" } } */
