// EXPECT-EXIT: 0
// _Float16 is IEEE binary16: 2 bytes, arithmetic rounds to half, f16 literal suffix.
extern void abort(void);
volatile _Float16 a = 1.0f16, b = 2.5f16;
_Float16 g = 0.1f16;
int main(void) {
    volatile _Float16 r = a + b;
    if (sizeof(_Float16) != 2 || r != 3.5f16) abort();
    if ((_Float16)2049.0 != 2048.0f16) abort();       /* ties-to-even at half precision */
    if (b * 3.0f16 != 7.5f16 || (int)(b * 2) != 5) abort();
    if (g == 0.1 || g != (_Float16)0.1) abort();         /* stored value is the half rounding */
    return 0;
}
