// EXPECT-EXIT: 0
// EXPECT-STDOUT: smashed
// A function with a local array gets a canary; overflowing the array must
// reach __stack_chk_fail before the function returns.
extern int printf (const char *, ...);
extern void exit (int);
void __stack_chk_fail (void) {
  printf ("smashed\n");
  exit (0);
}
__attribute__((noinline)) int smash (int n) {
  char buf[8];
  for (int i = 0; i < n; i++)
    buf[i] = 42;
  return 1;
}
int main (void) {
  smash (200);
  return 1;
}
