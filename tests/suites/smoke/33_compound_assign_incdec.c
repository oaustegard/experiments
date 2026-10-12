int printf(const char *fmt, ...);
int main(void) { int a = 10; a += 5; a -= 3; a *= 2; a /= 4; a %= 4; a <<= 3; a >>= 1; a |= 1; a &= 7; a ^= 2; int b = a++; b += ++a; printf("%d %d\n", a, b); return 0; }
