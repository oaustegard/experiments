// EXPECT-EXIT: 0
int x = 3;
int *q;
int fp(int *p) { q = p; return 1; }
void g(int a, int b[fp((int[2]){a, a + 2})]) {
  if (q[0] != a || q[1] != a + 2) __builtin_abort();
}
void h(int a[(int){x}]);
int main(void) {
  int t[1] = {0};
  g(1, t);
  return 0;
}
