/* { dg-do compile } */
/* { dg-options "-O2 -ftree-cselim -fdump-tree-phiopt1-details" } */

void f (int*);

void g3 (int i)
{
  int x = 0;
  if (i)
    x = i;
  f (&x);
}

/* { dg-final { scan-tree-dump "Conditional store replacement happened" "phiopt1" } } */
