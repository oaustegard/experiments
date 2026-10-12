// EXPECT-EXIT: 0
// sizeof(vla-expr) and __typeof__(vla-expr) evaluate the operand; plain
// sizeof/typeof of non-VM types do not.
static int i;
int foo(int n) {
  int (*t)[n];
  char b[1][n + 3];
  i = 0;
  sizeof(b[i++]);
  if (i != 1) return 1;
  __typeof__(t + (i++, 0)) p;
  if (i != 2) return 2;
  __typeof__(i + (i++, 0)) q;
  if (i != 2) return 3;
  sizeof(__typeof__(t + (i++, 0)));
  if (i != 2) return 4;
  return 0;
}
int main(void) { return foo(4); }
