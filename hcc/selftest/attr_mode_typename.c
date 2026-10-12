// EXPECT-EXIT: 0
// __attribute__((mode(QI))) in a type name (cast / sizeof) re-sizes the base type,
// as it does in a declaration.
extern void abort(void);
#define small __attribute__((mode(QI))) int
int main(void) {
  int x, y = 0x400;
  x = (small) y;
  if (sizeof(small) != sizeof(char))
    abort();
  if (sizeof(x) != sizeof(char) && x == y)
    abort();
  return 0;
}
