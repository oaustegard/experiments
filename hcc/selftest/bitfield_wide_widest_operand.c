// EXPECT-EXIT: 0
// Wide bit-field arithmetic is truncated to the widest operand's width
// (here 40), so 2^33 + 2^32 stays exact in the 40-bit result.
struct s { unsigned long long u33 : 33; unsigned long long u40 : 40; };
struct s b = { 0x100000000ULL, 0x100000000ULL };
int main(void) {
    if (b.u33 + b.u40 != 0x200000000ULL) return 1;
    return 0;
}
