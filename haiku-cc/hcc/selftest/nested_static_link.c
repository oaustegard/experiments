// EXPECT-EXIT: 0
// Block-scope function definitions (GNU nested functions) reading and
// writing enclosing locals, through a sibling and through a deeper level.
int foo(int i) {
  int a;
  void nested2(int i) { a = i; }
  void nested1(int i) {
    int b[32];
    for (int j = 0; j < 32; j++)
      b[j] = i + j;
    nested2(b[i]);
  }
  nested1(i);
  return a;
}
int f(int x) {
  int g(int y) { return y + x; }
  int h(int z) { int k(void) { return z * x; } return k() + g(1); }
  return g(1) + h(2);
}
int main(void) {
  if (foo(4) != 8) return 1;
  if (f(3) != 14) return 2;
  return 0;
}
