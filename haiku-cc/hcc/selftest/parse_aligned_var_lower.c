// EXPECT-EXIT: 0
// An aligned(N) attribute on a variable sets its alignment exactly, even lower.
int c __attribute__((aligned(2)));
_Static_assert(_Alignof(c) == 2, "aligned(2) lowers global alignment");
int main(void) {
    int l __attribute__((aligned(1)));
    _Static_assert(_Alignof(l) == 1, "aligned(1) lowers local alignment");
    return 0;
}
// packed member with an explicit aligned(N) keeps N as its alignment
struct P { char c; int a __attribute__((packed, aligned(2))); } p;
_Static_assert(_Alignof(p.a) == 2, "packed+aligned member");
_Static_assert(_Alignof(struct P) == 2, "packed+aligned member struct");
