// EXPECT-EXIT: 0
/* __int128 passed and returned in register pairs, and on the stack. */
typedef unsigned __int128 u128;
typedef __int128 i128;
static int bad;
#define CHECK(c) do { if (!(c)) bad++; } while (0)

static u128 twice(u128 x) { return x + x; }
static long pick(long k, u128 v, long j) { return k + (long)v + j; }
static i128 many(int a, i128 b, int c, i128 d, int e, int f, int g, i128 h, int i) {
  return b - d + h + (i128)(a + c + e + f + g + i);
}

int main(void) {
  u128 v = ((u128)1 << 65) + 7;
  CHECK(twice(v) == (((u128)4) << 64) + 14);
  CHECK(pick(1, v, 2) == 10);
  i128 m = many(1, (i128)1 << 70, 2, -((i128)1 << 80), 3, 4, 5, (i128)1 << 90, 6);
  CHECK(m == (((i128)1 << 90) + ((i128)1 << 80) + ((i128)1 << 70) + 21));
  return bad;
}
