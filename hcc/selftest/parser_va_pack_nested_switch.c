// EXPECT-EXIT: 3
// Nested switches with case labels inside if-statements, inside a function that
// uses __builtin_va_arg_pack(): the AST shares Case nodes between Switch.cases and
// the switch body; the pack pass must visit them once (it used to take 2^depth).
static int pick(int v) {
  switch (v) case 1: if (v) switch (v) case 2: if (v) switch (v) case 3: if (v)
  switch (v) case 4: if (v) switch (v) case 5: if (v) switch (v) case 6: if (v)
  switch (v) case 7: if (v) switch (v) case 8: if (v) switch (v) case 9: if (v)
  switch (v) case 10: if (v) switch (v) case 11: if (v) switch (v) case 12: if (v)
  switch (v) case 13: if (v) switch (v) case 14: if (v) switch (v) case 15: if (v)
  switch (v) case 16: if (v) switch (v) case 17: if (v) switch (v) case 18: if (v)
  switch (v) case 19: if (v) switch (v) case 20: if (v) switch (v) case 21: if (v)
  switch (v) case 22: if (v) switch (v) case 23: if (v) switch (v) case 24: if (v)
  switch (v) case 25: if (v) switch (v) case 26: if (v) switch (v) case 27: if (v)
  switch (v) case 28: if (v) switch (v) case 29: if (v) switch (v) case 30: if (v)
  switch (v) case 31: if (v) switch (v) case 32: if (v) return 1;
  return 3;
}
static int __attribute__((always_inline)) wrap(int a, ...) {
  return pick(a) + __builtin_va_arg_pack_len() * 0;
}
int main(void) { return wrap(0); }
