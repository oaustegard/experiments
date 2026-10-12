// EXPECT-EXIT: 0
// K&R parameter declarations may carry trailing attributes: `int a __attribute__((unused));`
#define UNUSED __attribute__((unused))
int f(a, b)
  int a UNUSED;
  int b __attribute__((unused));
{
  return a + b;
}
int main(void) {
  return f(1, 2) == 3 ? 0 : 1;
}
