// EXPECT-EXIT: 0
typedef struct { char c[8]; } V __attribute__((aligned(16)));
typedef __attribute__((aligned(32))) int I;
V v;
I i;
int main(void) {
    if (((unsigned long)&v) % 16) return 1;
    if (((unsigned long)&i) % 32) return 2;
    return 0;
}
