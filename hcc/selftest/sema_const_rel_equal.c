// EXPECT-EXIT: 0
extern void link_error(void);
static int ok = 0;
static void test(int a, int b) {
    if ((a < b) != (b > a)) link_error();
    if ((a - 1 < b) != (a <= b)) link_error();
    if ((a + -2 >= b) != (a + -1 > b)) link_error();
    ok = 1;
}
int main(void) {
    test(1, 2);
    if (__builtin_bswap16(0xabcd) != 0xcdab) link_error();
    if (__builtin_bswap32(0x89abcdef) != 0xefcdab89) link_error();
    if (__builtin_bswap64(0x0123456789abcdefULL) != 0xefcdab8967452301ULL) link_error();
    return ok ? 0 : 1;
}
