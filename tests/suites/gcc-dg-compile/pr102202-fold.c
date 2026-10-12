/* PR tree-optimization/102202 */
/* { dg-do compile } */
/* { dg-options "-O2 -fdump-tree-optimized" } */

/* A constant-length-one memset is folded to a single byte store, even for a
   store through an arbitrary pointer.  */

void *
g1 (char *d, int c)
{
  return __builtin_memset (d, c, 1);
}

void
g2 (char *d, int c)
{
  __builtin_memset (d, c, 1);
}

void *
g3 (int *p, int c)
{
  return __builtin_memset (p, c, 1);
}

void *
g4 (char *d)
{
  return __builtin_memset (d, 7, 1);
}

/* { dg-final { scan-tree-dump-not "__builtin_memset" "optimized" } } */
/* { dg-final { scan-tree-dump-times {\(unsigned char\) c_[0-9]+\(D\)} 3 "optimized" } } */
/* { dg-final { scan-tree-dump-times {MEM[^;\n\r]*= _[0-9]+;} 3 "optimized" } } */
/* { dg-final { scan-tree-dump-times {MEM[^;\n\r]*= 7;} 1 "optimized" } } */
