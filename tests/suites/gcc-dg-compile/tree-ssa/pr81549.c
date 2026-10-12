/* PR tree-optimization/81549 */
/* { dg-do compile } */
/* { dg-options "-O2 -fdump-tree-sccp-details -fdump-tree-optimized" } */

int a[10000];

int
f (void)
{
  int t0 = a[0];
  int t1 = a[1];

  for (int i = 0; i < 100; ++i)
    {
      a[i] = 1;
      t0 = t1;
      t1 = 2;
    }

  a[100] = t0;
  a[101] = t1;
  return t0 + t1;
}

/* { dg-final { scan-tree-dump-times "with expr: 2" 1 "sccp" } } */
/* { dg-final { scan-tree-dump {a\[100\] = 2;} "sccp" } } */
/* { dg-final { scan-tree-dump {a\[101\] = 2;} "sccp" } } */
/* { dg-final { scan-tree-dump "return 4;" "optimized" } } */
