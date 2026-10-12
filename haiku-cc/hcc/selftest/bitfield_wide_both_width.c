// EXPECT-EXIT: 0
// Both operands bit-fields: the result is truncated to the widest width (40),
// a 64-bit operand (ULL constant) disables the narrowing.
struct s { unsigned long long a:33; unsigned long long b:40; };
struct s t = {0x100000, 0x100000000ULL};
int main(void) {
    if (t.a - t.b != 0xFF00100000ULL) return 1;
    if (t.a * t.a != 0) return 2;
    if (t.a - 0x100000000ULL != 0xFFFFFFFF00100000ULL) return 3;
    return 0;
}
