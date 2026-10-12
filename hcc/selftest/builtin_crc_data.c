// EXPECT-EXIT: 0
// GCC CRC builtins: chained byte-wise CRC-8 (SMBUS, poly 0x07) and reflected CRC-32
// over "123456789" must give the standard check values 0xF4 and 0xCBF43926.
// A user function named crc8_data8 must not hijack the builtin.
#include <stdint.h>
static uint8_t crc8_data8(uint8_t x) { return x; }
int main(void) {
    const char *s = "123456789";
    uint8_t c = 0;
    uint32_t r = 0xFFFFFFFFu;
    for (int i = 0; s[i]; i++) {
        c = __builtin_crc8_data8(c, (uint8_t)s[i], 0x07);
        r = __builtin_rev_crc32_data8(r, (uint8_t)s[i], 0xEDB88320u);
    }
    if (c != 0xF4) return 1;
    if ((r ^ 0xFFFFFFFFu) != 0xCBF43926u) return 2;
    if (crc8_data8(7) != 7) return 3;
    return 0;
}
