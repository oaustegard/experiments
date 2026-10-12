// EXPECT-EXIT: 0
// EXPECT-FLAGS: -std=c23
// `constexpr T *p` is `T *const p`: the qualifier applies to the pointer, and the
// pointee type keeps its own qualifiers (no const added to the target).
constexpr int *restrict *v13 = 0;
typedef int *P;
constexpr volatile int *v11 = 0;
extern int *const *cp;
extern volatile int *const *cpv;
int *const *cp = 0;
volatile int *const *cpv = 0;
int main (void) {
    int *x = 0;
    int *const *px = &x;
    int *y = *px;
    return (y != 0) + (v11 != 0) + (v13 != 0);
}
