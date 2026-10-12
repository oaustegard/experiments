// EXPECT-EXIT: 0
// __builtin_call_with_static_chain(call, chain): the chain value is in %r10 during
// the call; a register-bound local (`register T x __asm__("r10")`) reads it at entry.
__attribute__((noinline)) void *get_chain(void) {
    register void *c __asm__("%r10");
    return c;
}
void *(*fp)(void) = get_chain;
int main(void) {
    char c;
    if (__builtin_call_with_static_chain(fp(), &c) != &c) return 1;
    if (__builtin_call_with_static_chain(get_chain(), &c) != &c) return 2;
    return 0;
}
