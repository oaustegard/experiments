/* PR tree-optimization/109878 */
/* { dg-do compile } */
/* { dg-options "-O1 -fdump-tree-optimized" } */

/* The testcases here should not get optimized with the patch.
   For constant pair <cst0, cst1>, the condition:
   (cst0 & cst1 == cst0) || (cst0 & cst1 == cst1)
   is false for the constants used here. */
int max_and(int a, int b) {

  b = a & 3;
  a = a & 5;
  if (b > a)
    return b;
  else
    return a;
}

/* The constants in this function satisfy the condition but a is signed.
   For signed types both the constants should have same sign. */
int min_and(int a, int b) {
  b = a & 1;
  a = a & -3;
  if (b < a)
    return b;
  else
    return a;
}

/* { dg-final { scan-tree-dump " MIN_EXPR " "optimized" } } */
/* { dg-final { scan-tree-dump " MAX_EXPR " "optimized" } } */
