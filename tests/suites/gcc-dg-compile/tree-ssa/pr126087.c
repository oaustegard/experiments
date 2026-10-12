/* PR tree-optimization/126087 */
/* { dg-do compile } */
/* { dg-options "-O2 -fdump-tree-optimized" } */

void use();

void f2(int a, unsigned short bb) {
    int b = bb;
    int m = a < b ? a : b;
    if (m >= 0)
      use();
}

void f3(int a, unsigned short bb) {
    int b = bb;
    int m = a | b;
    if (m >= 0)
      use();
}

void f4(int a, unsigned short bb) {
    int b = bb;
    int m = a < b ? a : b;
    if (m < 0)
      use();
}

void f5(int a, unsigned short bb) {
    int b = bb;
    int m = a | b;
    if (m < 0)
      use();
}

/* { dg-final { scan-tree-dump-not "MIN_EXPR" "optimized" } } */
/* { dg-final { scan-tree-dump-not " \\| " "optimized" } } */
