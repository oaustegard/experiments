// EXPECT-EXIT: 0
// C23 `(static thread_local int []){...}` / `(thread_local ...)`: a compound
// literal with thread storage duration is per-thread, but its address is
// stable within one thread across calls.
int *get(void) {
    return (static _Thread_local int []) { 1, 2 };
}
int main(void) {
    int *p = get();
    if (p[0] != 1 || p[1] != 2) return 1;
    p[0] = 7;
    if (get() != p) return 2;
    return 0;
}
