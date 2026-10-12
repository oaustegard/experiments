// EXPECT-EXIT: 0
void *f(int n) {
    return __builtin_alloca_with_align(n, 128);
}
int main(void) {
    for (int i = 1; i < 40; i += 7) {
        unsigned long p = (unsigned long)f(i);
        if (p & 15) return 1;
    }
    return 0;
}
