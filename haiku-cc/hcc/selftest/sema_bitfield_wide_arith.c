// EXPECT-EXIT: 0
struct wide { unsigned long long b : 40; } x;
struct sw { long long s : 40; } y;
int main(void) {
  x.b = 0x0100000001ULL;
  if ((x.b << 8) != 0x100) return 1;
  if ((x.b << 8) + (x.b >> 32) != 0x101) return 2;
  x.b = 0xFF00000001ULL;
  if (x.b + x.b != 0xFE00000002ULL) return 3;
  y.s = 0x7FFFFFFFFFLL;
  if (y.s + 1 != -0x8000000000LL) return 4;
  return 0;
}
