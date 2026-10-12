// EXPECT-EXIT: 0
// Packed bitfield wider than the remaining bits of its type's storage unit
// (unsigned b:32 at bit offset 6) must initialize and read back without overflow.
struct __attribute__((packed)) S {
    unsigned a : 6;
    unsigned b : 32;
    unsigned c : 2;
};
struct S s = { 2, 0x12345678, 1 };
int main(void) {
    return (s.a == 2 && s.b == 0x12345678 && s.c == 1) ? 0 : 1;
}
