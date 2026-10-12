/* { dg-do compile } */
/* { dg-options "-O1 -fdump-tree-phiopt1" } */

int t;
void f(int a, int b, int c, int d)
{
  if (b)
   t = 0;
  else
   t = 0;
}

/* We should sink/merge the stores and end up with a single BB.  */

/* { dg-final { scan-tree-dump-times "t = 0;" 1 "phiopt1" } } */
/* { dg-final { scan-tree-dump-times "<bb " 1 "phiopt1" } } */
