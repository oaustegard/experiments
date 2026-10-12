/* { dg-options "-O1 -fdump-tree-original" } */

int gg (unsigned u, _Bool b)
{
  return u <= (u <= (unsigned)b);
}

int gg2 (unsigned u, _Bool b)
{
  return ((unsigned)b >= u) >= u;
}

int gg3 (unsigned u, _Bool b)
{
  return u > (u <= (unsigned)b);
}

int gg4 (unsigned u, _Bool b)
{
  return ((unsigned)b >= u) < u;
}

/* { dg-final { scan-tree-dump-times " >= " 2 "original" } } */
/* { dg-final { scan-tree-dump-times " < " 2 "original" } } */
