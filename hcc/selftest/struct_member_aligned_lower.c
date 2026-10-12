// EXPECT-EXIT: 0
// aligned(N) on a struct member must not lower its natural alignment.
struct A { char c; long long i; };
struct B { char c; long long i __attribute__((__aligned__(1))); };
int main(void) {
    if (sizeof(struct A) != sizeof(struct B)) return 1;
    if (_Alignof(struct B) != 8) return 2;
    return 0;
}
