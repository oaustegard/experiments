/* PR tree-optimization/109878 */
/* { dg-do compile } */
/* { dg-options "-O1 -fdump-tree-optimized" } */

/* For unsigned types, min(a, a&CST) should be simplified to a&CST and
   should not generate MIN_EXPR. */
unsigned int min_1(unsigned int a, unsigned int b) {
  b = a & 1;
  if (b < a)
    return b;
  else
    return a;
}

unsigned int min_2(unsigned int a, unsigned int b) {
  b = a & 3;
  if (b < a)
    return b;
  else
    return a;
}

/* For unsigned types, max(a, a&CST) should be simplified to a and
   should not generate MAX_EXPR. */
unsigned int max_1(unsigned int a, unsigned int b) {
  b = a & 1;
  if (b > a)
    return b;
  else
    return a;
}

unsigned int max_2(unsigned int a, unsigned int b) {
  b = a & 3;
  if (b > a)
    return b;
  else
    return a;
}

/* { dg-final { scan-tree-dump-not " MIN_EXPR " "optimized" } } */
/* { dg-final { scan-tree-dump-not " MAX_EXPR " "optimized" } } */
