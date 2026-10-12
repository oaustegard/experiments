/* { dg-do compile } */
/* { dg-options "-O1 -fdump-tree-phiopt1" } */

/* Test that (A > B) ? ABS(A) : B is simplified to MAX(A, B)
   when B is known to be non-negative. */

int f(unsigned char a, int b, int c)
{
  int t = a;
  if (c > t)
    t = (c > 0 ? c : -c);
  return t;
}

int f1(unsigned char a, int b, int c)
{
  int t = a;
  if (c > t) return (c > 0 ? c : -c);
  return t;
}

/* Both functions should be converted to MAX_EXPR by phiopt1.  */
/* { dg-final { scan-tree-dump-times "MAX_EXPR" 2 "phiopt1" } } */
/* { dg-final { scan-tree-dump-not "ABS_EXPR" "phiopt1" } } */
/* { dg-final { scan-tree-dump-not "if " "phiopt1" } } */
