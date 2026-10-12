// EXPECT-ERROR: [parse]
// __atomic_fetch_add / __sync_fetch_and_add on a _Bool pointer must be rejected (gcc PR 68966).
void f(_Bool *b) {
    __atomic_fetch_add(b, 1, 5);
}
