// EXPECT-EXIT: 0
struct s { unsigned long long a:2; unsigned long long b:40; unsigned long long c:22; };
int main(void) {
    struct s t = {1, 2, 3};
    unsigned long long x = t.b - 8;
    if (x != 0xFFFFFFFFFAULL) return 1;
    if (t.b * 3 != 6) return 2;
    return 0;
}
