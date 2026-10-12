// EXPECT-EXIT: 0
int foo(unsigned long long v) {
    switch (v) {
    case 1000000000000000000ULL ... 9999999999999999999ULL: return 19;
    case 5 ... 7: return 3;
    default: return 20;
    }
}
int main(void) {
    if (foo(1000000000000000000ULL) != 19) return 1;
    if (foo(6) != 3) return 2;
    if (foo(8) != 20) return 3;
    return 0;
}
