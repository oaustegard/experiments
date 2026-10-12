// EXPECT-EXIT: 0
typedef int T;
int main(void) {
    if (sizeof(enum { T }) == 0) return 1;
    T x = 3;
    if (sizeof(struct S { int a; }) != 4) return 2;
    return x == 3 ? 0 : 3;
}
