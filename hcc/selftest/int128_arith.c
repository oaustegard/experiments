// EXPECT-EXIT: 0
/* __int128 arithmetic, shifts, comparisons, conversions (values checked against gcc). */
typedef unsigned __int128 u128;
typedef __int128 i128;
#define U(hi, lo) ((((u128)(hi)) << 64) | (u128)(lo))

static int bad;
#define CHECK(c) do { if (!(c)) bad++; } while (0)

int main(void) {
  u128 a = ((u128)1 << 100) + 12345, b = 987654321ull;
  CHECK(a + b == U(0x1000000000ull, 0x3ade98eaull));
  CHECK(a - b == U(0xfffffffffull, 0xffffffffc521c788ull));
  CHECK(a * b == U(0xade68b1000000000ull, 0xb16cf267f69ull));
  CHECK(a / b == U(0x45ull, 0x94169f665bf926daull));
  CHECK(a % b == 0x1be7c37full);
  CHECK((a >> 7) == U(0x20000000ull, 0x60ull));
  CHECK((a << 70) == U(0xc0e40ull, 0));
  CHECK((a >> 70) == U(0, 0x40000000ull));
  CHECK(!(a < b) && (a > b) && (a == a) && !(a != a));
  u128 g = 5;
  g = ~g;
  g += 1;
  CHECK(g == U(0xffffffffffffffffull, 0xfffffffffffffffbull));
  i128 s = -((i128)1 << 90);
  CHECK((u128)(s >> 80) == U(0xffffffffffffffffull, 0xfffffffffffffc00ull));
  CHECK((u128)-s == U(0x4000000ull, 0));
  CHECK(s < 0 && !(s > 0));
  CHECK((i128)-7.5e20 >> 64 == -41);
  CHECK((double)a == 1267650600228229401496703205376.0);
  CHECK((int)(a & 0xff) == 57);
  CHECK(!(u128)0 && !!a);
  return bad;
}
