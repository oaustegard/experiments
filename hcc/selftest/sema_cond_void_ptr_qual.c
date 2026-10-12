// EXPECT-EXIT: 0
// C23: `cond ? T(*)[N] : void *` keeps the array element qualifiers (const),
// and _Atomic is not carried into the void pointer result.
int main(void) {
    const int (*u)[1] = 0;
    void *v = 0;
    _Atomic void *w = 0;
    _Static_assert(_Generic(1 ? u : v, const void *: 1, default: 0), "lost qualifier");
    _Static_assert(_Generic(0 ? v : u, const void *: 1, default: 0), "lost qualifier");
    _Static_assert(_Generic(1 ? w : v, void *: 1, default: 0), "_Atomic should be removed");
    _Static_assert(_Generic(0 ? v : w, _Atomic void *: 0, void *: 1), "_Atomic should be removed");
    return 0;
}
