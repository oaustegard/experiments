/* { dg-do compile } */
/* { dg-options "-O1 -fdump-tree-optimized" } */

int
bit_trick (int x)
{
  int mask = -(x < 0);
  return ((x + mask) ^ mask);
}

/* { dg-final { scan-tree-dump "ABS_EXPR" "optimized" } } */
