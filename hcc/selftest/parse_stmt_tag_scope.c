// EXPECT-EXIT: 0
// Tags declared in a controlling expression (sizeof type name) are local to the
// if/for/while statement and its substatements; a tag in a substatement may be
// redeclared without a redefinition error (C11 6.8.4p3, 6.8.5p5).
extern void abort(void);
int main(void) {
  struct foo { int i0; };
  int a = sizeof(struct foo), b = 0, c = 0, d = 1;
  if (b = sizeof(struct foo { int i0; int i1; }))
    c = sizeof(struct foo { int i0; int i1; int i2; });
  if (!(a <= b && b <= c))
    abort();
  for (; d; d--)
    c = sizeof(struct foo);
  if (c != a)
    abort();
  return 0;
}
