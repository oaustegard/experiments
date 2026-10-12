// EXPECT-ERROR: [parse] missing loop condition in loop with 'GCC ivdep' pragma
// #pragma GCC ivdep needs a loop condition: a for loop without one is an error.
int main(void) {
  int i;
#pragma GCC ivdep
  for (i = 0; ; ++i) { }
  return 0;
}
