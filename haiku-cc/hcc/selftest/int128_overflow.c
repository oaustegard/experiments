// EXPECT-EXIT: 0
/* __builtin_{add,sub}_overflow[_p] with __int128 operands and results. */
typedef unsigned __int128 u128;
typedef __int128 i128;
static int bad;
#define CHECK(c) do { if (!(c)) bad++; } while (0)

int main(void) {
  u128 m = ~(u128)0, r;
  i128 s = -1, t;
  unsigned long long ul = ~0ull;
  CHECK(__builtin_add_overflow_p(m, 0, (u128)0) == 0);
  CHECK(__builtin_add_overflow_p(m, 1, (unsigned long)0) == 1);
  CHECK(__builtin_sub_overflow_p((u128)0, 1, (u128)0) == 1);
  CHECK(__builtin_add_overflow_p(s, 1, (i128)0) == 0);
  CHECK(__builtin_add_overflow_p((i128)1 << 126, (i128)1 << 126, (i128)0) == 1);
  CHECK(__builtin_add_overflow(ul, ul, &r) == 0 && r == (u128)ul * 2);
  CHECK(__builtin_add_overflow(m, 2, &r) == 1 && r == 1);
  CHECK(__builtin_sub_overflow(m, m, &r) == 0 && r == 0);
  CHECK(__builtin_sub_overflow(s, (i128)1 << 126, &t) == 0 && t == -1 - ((i128)1 << 126));
  CHECK(__builtin_add_overflow((i128)1 << 126, (i128)1 << 126, &t) == 1 && t == -((i128)1 << 127));
  long long small;
  CHECK(__builtin_add_overflow(s, 5, &small) == 0 && small == 4);
  return bad;
}
