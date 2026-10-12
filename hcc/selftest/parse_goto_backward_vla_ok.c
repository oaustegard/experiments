// EXPECT-EXIT: 0
int f(int n) {
  int r = 0;
  goto a;
b:
  { int c[n]; c[0] = 1; r += c[0]; }
  goto done;
a:
  { int d[n]; d[0] = 2; r += d[0]; goto b; }
done:
  return r;
}
int main(void) { return f(3) == 3 ? 0 : 1; }
