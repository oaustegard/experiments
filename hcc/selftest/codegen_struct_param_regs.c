// EXPECT-EXIT: 0
// A small struct by-value param must not clobber later GP argument registers.
struct tiny { char c; };
int f(int n, struct tiny x, struct tiny y, struct tiny z, long l) {
  if (x.c != 10 || y.c != 11 || z.c != 12 || l != 123) return 1;
  return 0;
}
int main(void) {
  struct tiny x[3];
  x[0].c = 10; x[1].c = 11; x[2].c = 12;
  return f(3, x[0], x[1], x[2], (long)123);
}
