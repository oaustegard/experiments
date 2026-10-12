// EXPECT-EXIT: 0
// __builtin_assume_aligned yields its pointer argument; extra arguments are evaluated.
static int hits;
static int bump(void) { hits++; return 2; }
__attribute__((noinline)) unsigned char *al(unsigned char *p) { return __builtin_assume_aligned(p, bump()); }
__attribute__((noinline)) unsigned char *al3(unsigned char *p) { return __builtin_assume_aligned(p, 8, 0); }
int main(void) {
  unsigned char b[4] = {1, 2, 3, 4};
  if (al(b)[1] != 2) return 1;
  if (al3(b)[2] != 3) return 2;
  if (hits != 1) return 3;
  return 0;
}
