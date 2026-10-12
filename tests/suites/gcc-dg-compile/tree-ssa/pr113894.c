/* PR tree-optimization/113894 */
/* { dg-do compile } */
/* { dg-options "-O2 -fdump-tree-forwprop1" } */

int f_cmp_lt(int x, int y)
{
  int cmp = x < y;
  return (x ^ -cmp) + cmp;
}

int f_cmp_gt_commuted(int x, int y)
{
  int cmp = x > y;
  return cmp + (-cmp ^ x);
}

unsigned f_unsigned_cmp(unsigned x, unsigned y)
{
  unsigned cmp = x < y;
  return (x ^ -cmp) + cmp;
}

int f_mask(int x, unsigned y)
{
  int cmp = y & 1;
  return (x ^ -cmp) + cmp;
}

int f_bool(int x, _Bool cmp)
{
  int icmp = cmp;
  return (x ^ -icmp) + icmp;
}

int f_abs_int(int x)
{
  int cmp = x < 0;
  return (x ^ -cmp) + cmp;
}

long f_abs_long(long x)
{
  long cmp = x < 0;
  return (x ^ -cmp) + cmp;
}

int f_signed_not_zero_one(int x, int cmp)
{
  return (x ^ -cmp) + cmp;
}

unsigned f_unsigned_not_zero_one(unsigned x, unsigned cmp)
{
  return (x ^ -cmp) + cmp;
}

/* The branchless conditional negate spelling should only survive when cmp is
   not known to be 0 or 1.  */
/* { dg-final { scan-tree-dump-times " \\^ " 2 "forwprop1" } } */
/* Sign tests should expose absolute value.  */
/* { dg-final { scan-tree-dump-times " = ABS_EXPR" 2 "forwprop1" } } */
/* Other zero-one predicates should expose conditional negation.  */
/* { dg-final { scan-tree-dump-times " \\? " 5 "forwprop1" } } */
