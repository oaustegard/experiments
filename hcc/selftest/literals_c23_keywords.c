// EXPECT-EXIT: 0
// EXPECT-FLAGS: -std=c23
// C23 makes alignas/alignof/static_assert/thread_local keywords.
_Static_assert(1, "ok");
static_assert(sizeof(int) == 4, "int is 4 bytes");
#define MY_ALIGNOF alignof
int main(void) {
    alignas(16) static char buf[4];
    static thread_local int tl = 0;
    return (int)MY_ALIGNOF(int) - 4 + (int)(_Alignof(char) - 1) + tl;
}
