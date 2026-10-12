// EXPECT-EXIT: 0
typedef unsigned __int128 U128 __attribute__((vector_size(32)));
int main(void) {
    U128 a = {((unsigned __int128)1 << 64) + 1, 3};
    U128 b = {((unsigned __int128)1 << 64) + 1, 5};
    U128 c = a * b;
    /* (2^64+1)^2 = 2^128 + 2^65 + 1 == 2^65 + 1 (mod 2^128); 3*5 = 15 */
    if (c[0] != (((unsigned __int128)1 << 65) + 1)) return 1;
    if (c[1] != 15) return 2;
    return 0;
}
