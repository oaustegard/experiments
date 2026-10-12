/* PR tree-optimization/109878 */
/* { dg-do compile } */
/* { dg-options "-O1 -fdump-tree-optimized" } */

/* All the constant pair <cst0, cst1> used here satisfy the condition:
   (cst0 & cst1 == cst0) || (cst0 & cst1 == cst1).
   If the above condition is true, then MIN_EXPR is not needed. */
int min_and(int a, int b) {
  b = a & 3;
  a = a & 1;
  if (b < a)
    return b;
  else
    return a;
}

int min_and1(int a, int b) {
  b = a & 3;
  a = a & 15;
  if (b < a)
    return b;
  else
    return a;
}

int min_and2(int a, int b) {
  b = a & -7;
  a = a & -3;
  if (b < a)
    return b;
  else
    return a;
}

int min_and3(int a, int b) {
  b = a & -5;
  a = a & -13;
  if (b < a)
    return b;
  else
    return a;
}

/* When constants are of opposite signs, the simplification will only
   work for unsigned types. */
unsigned int min_and4(unsigned int a, unsigned int b) {
  b = a & 3;
  a = a & -5;
  if (b < a)
    return b;
  else
    return a;
}

unsigned int min_and5(unsigned int a, unsigned int b) {
  b = a & -3;
  a = a & 5;
  if (b < a)
    return b;
  else
    return a;
}

/* { dg-final { scan-tree-dump-not " MIN_EXPR " "optimized" } } */
