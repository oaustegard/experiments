/* PR middle-end/122715 */
/* { dg-do compile } */
/* { dg-options "-O2 -fgimple -fdump-tree-optimized" } */

typedef unsigned int v4u32 __attribute__((vector_size(16)));

v4u32 __GIMPLE() avg_floor(v4u32 x)
{
  v4u32 res;
  res_1 = .AVG_FLOOR (x, x);
  return res_1;
}

v4u32 __GIMPLE() avg_ceil(v4u32 x)
{
  v4u32 res;
  res_1 = .AVG_CEIL (x, x);
  return res_1;
}

/* { dg-final { scan-tree-dump-not {\.AVG_} "optimized" } } */
/* { dg-final { scan-tree-dump-times {return x_} 2 "optimized" } } */
