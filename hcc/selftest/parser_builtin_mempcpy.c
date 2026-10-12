// EXPECT-EXIT: 0
// EXPECT-STDOUT: 1
// __builtin_mempcpy without a prototype returns void *, not an implicit int.
struct X { int *p; int *q; };
int foo (void) {
  int i = 0, j = 1;
  struct X x, y;
  int **p;
  y.p = &i;
  x.q = &j;
  p = __builtin_mempcpy (&x, &y, sizeof (int *));
  return **p;
}
extern int printf (const char *, ...);
int main (void) {
  printf ("%d\n", foo ());
  return 0;
}
