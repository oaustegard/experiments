/* { dg-do compile } */
/* { dg-options "-Ofast -fdump-tree-forwprop1-raw" } */

/* "min|max(a+|-c,b+|-c) -> min|max(a,b)+|-c" folding should NOT happen for
   non-integral types.  */

double
f (double a, double b, double c)
{
  return __builtin_fmax (a + c, b + c);
}

/* { dg-final { scan-tree-dump-times "plus_expr" 2 "forwprop1" } } */
