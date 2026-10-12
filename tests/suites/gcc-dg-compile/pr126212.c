/* { dg-do compile } */
/* { dg-options "-O2 -fdump-tree-evrp" } */

extern void keep (void);
extern void kill (void);
void
f (void *p, void *q, int flag, int x)
{
  int eq = p == q;
  int nz = flag != 0;
  int both = eq & nz;

  if (both)
    return;

  if (!nz)
    return;

  if (p == q)
    kill ();
  else
    keep ();
}

/* { dg-final { scan-tree-dump-not "kill" "evrp" } } */
