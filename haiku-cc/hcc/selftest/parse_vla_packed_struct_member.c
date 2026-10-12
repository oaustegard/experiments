// EXPECT-EXIT: 0
extern void abort(void);
int f(int n) {
  struct bar { unsigned char a; int b[n]; } __attribute__((__packed__)) bar;
  struct foo { unsigned char a; int b[4]; } __attribute__((__packed__)) foo;
  if (sizeof(bar) != sizeof(foo)) abort();
  return 0;
}
int main(void) { return f(4); }
