// EXPECT-EXIT: 0
#define ENDIAN __attribute((scalar_storage_order ("big-endian")))
typedef struct ENDIAN { __uint128_t t:124; __uint128_t t1:4; } g;
int main(void) {
    g y = {0};
    y.t = 0x77;
    y.t1 = 9;
    if (y.t != 0x77) return 1;
    if (y.t1 != 9) return 2;
    return 0;
}
