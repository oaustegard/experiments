// EXPECT-EXIT: 0
struct f { __uint128_t t:124; __uint128_t t1:4; };
struct f mk(void) { struct f s = {1, 2}; return s; }
int main(void) {
    struct f x = mk();
    if ((unsigned long long)x.t != 1) return 1;
    if ((unsigned long long)x.t1 != 2) return 2;
    x.t = 0xdeadbeefULL;
    x.t1 = 9;
    if ((unsigned long long)x.t != 0xdeadbeefULL) return 3;
    if ((unsigned long long)x.t1 != 9) return 4;
    return 0;
}
