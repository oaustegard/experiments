// EXPECT-EXIT: 0
int f(int a, int b[*][*], int c[sizeof(*b)]);
int f(int a, int b[2][2], int c[4]) {
  return sizeof(c) == sizeof(int *) ? 0 : 1;
}
int main(void) {
  int b[2][2];
  int c[4];
  return f(0, b, c);
}
