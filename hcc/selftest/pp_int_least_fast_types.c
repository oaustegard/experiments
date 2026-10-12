// EXPECT-EXIT: 0
typedef __INT_LEAST8_TYPE__ least8_t;
typedef __UINT_LEAST16_TYPE__ uleast16_t;
typedef __INT_FAST16_TYPE__ fast16_t;
typedef __UINT_FAST32_TYPE__ ufast32_t;
int main(void) {
    least8_t a = (least8_t)-1;
    uleast16_t b = (uleast16_t)0xffff;
    fast16_t c = __INT_FAST16_MAX__;
    ufast32_t d = __UINT_FAST32_MAX__;
    if (a != -1) return 1;
    if (b != 0xffff) return 2;
    if (c != 0x7fffffffffffffffL) return 3;
    if (d != 0xffffffffffffffffUL) return 4;
    if (__INT_LEAST32_MAX__ != 0x7fffffff) return 5;
    return 0;
}
