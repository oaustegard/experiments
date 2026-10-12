// EXPECT-EXIT: 0
// double/float -> unsigned long must be exact for values in [2^63, 2^64).
extern void abort(void);
int main(void) {
    volatile double d = 1.5e19;
    volatile float f = 3.0e9f;
    unsigned long long a = (unsigned long long)d;
    unsigned long long b = (unsigned long long)f;
    if (a != 15000000000000000000ULL) abort();
    if (b != 3000000000ULL) abort();
    if ((unsigned long long)(volatile double)9.5e18 != 9500000000000000000ULL) abort();
    return 0;
}
