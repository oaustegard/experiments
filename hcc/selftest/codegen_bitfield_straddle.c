// EXPECT-EXIT: 0
// Packed 64-bit bitfield at bit 4 spans 9 bytes: read and write must use the extra byte.
struct __attribute__((packed)) s { unsigned char p : 4; unsigned long long a : 64; unsigned long long b : 60; };
int main(void) {
    struct s x;
    x.p = 15; x.a = ~0ULL; x.b = ~0ULL;
    unsigned long long *w = (unsigned long long *)&x + 1;
    if (*w != ~0ULL) return 1;
    x.a = 0x123456789abcdef0ULL;
    if (x.a != 0x123456789abcdef0ULL) return 2;
    if (x.p != 15 || x.b != ((1ULL << 60) - 1)) return 3;
    x.b = 0;
    if (x.a != 0x123456789abcdef0ULL || x.p != 15) return 4;
    return 0;
}
