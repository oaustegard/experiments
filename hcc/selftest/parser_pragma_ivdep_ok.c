// EXPECT-EXIT: 3
// #pragma GCC ivdep is accepted before a for loop with a condition.
int main(void) {
  int i, s = 0;
#pragma GCC ivdep
  for (i = 0; i < 3; ++i) s += i;
  return s;
}
