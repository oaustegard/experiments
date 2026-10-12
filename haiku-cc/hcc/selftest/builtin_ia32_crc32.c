// EXPECT-EXIT: 0
// SSE4.2 CRC32 builtins: the value operand width selects crc32b/w/l/q.
int main(void) {
    unsigned int a = __builtin_ia32_crc32qi(0xFFFFFFFFu, 0x31);
    unsigned int b = __builtin_ia32_crc32hi(0u, 0x3231);
    unsigned int c = __builtin_ia32_crc32si(0u, 0x34333231u);
    unsigned long long d = __builtin_ia32_crc32di(0ull, 0x3837363534333231ull);
    if (a != 0x6f0a661cu) return 1;
    if (b != 0x8234b3b2u) return 2;
    if (c != 0xbe5dbf29u) return 3;
    if (d != 0xecaf3210ull) return 4;
    return 0;
}
