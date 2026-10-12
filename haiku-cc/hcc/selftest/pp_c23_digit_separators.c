// EXPECT-EXIT: 0
// EXPECT-FLAGS: -std=c23
int main(void) {
    if (1'000 != 1000) return 1;
    if (0x0'e - 0xe != 0) return 2;
    if (0xe-1 != 13) return 3;
#line 4'56
    if (__LINE__ != 456) return 4;
    return 0;
}
