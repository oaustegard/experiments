/* The six-operation rounded (ceiling) average should collapse to the
   four-operation form, the dual of the floor rule:
     ((x >> 1) + (y >> 1)) + ((x | y) & 1)  ->  (x | y) - ((x ^ y) >> 1)
   x + y is 2 * (x | y) - (x ^ y), so halving the sum rounded up needs only
   an ior, an xor, a shift and a subtract.  */
/* { dg-do compile } */
/* { dg-options "-O2 -fdump-tree-optimized" } */

unsigned
avg_ceil (unsigned x, unsigned y)
{
  return ((x >> 1) + (y >> 1)) + ((x | y) & 1u);
}

int
avg_ceil_signed (int x, int y)
{
  return ((x >> 1) + (y >> 1)) + ((x | y) & 1);
}

/* Both shifts of the operands must be gone, leaving one shift of the xor.  */
/* { dg-final { scan-tree-dump-times " >> 1;" 2 "optimized" } } */
/* { dg-final { scan-tree-dump-times " \\^ " 2 "optimized" } } */
/* { dg-final { scan-tree-dump-times " \\| " 2 "optimized" } } */
/* { dg-final { scan-tree-dump-not " & 1;" "optimized" } } */
