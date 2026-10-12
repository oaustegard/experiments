/* PR tree-optimization/116700 */
/* { dg-do compile } */
/* { dg-options "-O1 -fdump-tree-phiopt1" } */

int f(unsigned char a, int b, int c)
{
  int t = a;
  if (c > t) t = (c > 0 ? c : -c);
  return t;
}

/* { dg-final { scan-tree-dump "MAX_EXPR" "phiopt1" } } */
/* { dg-final { scan-tree-dump-not "ABS_EXPR" "phiopt1" } } */
