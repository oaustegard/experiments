// EXPECT-EXIT: 0
/* Shifts on 16-byte (__int128) vector lanes: << and >> with a small count. */
typedef __int128 V __attribute__((vector_size(32)));
int main(void) {
    V a = {12, 0};
    V b = {3, 0};
    if ((a << b)[0] != 96) return 1;
    if ((a >> b)[0] != 1) return 2;
    return 0;
}
