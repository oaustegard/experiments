// EXPECT-EXIT: 0
void* alloc_a(int) __attribute__((malloc(__builtin_free)));
void* alloc_b(int) __attribute__((malloc(__builtin_free)));
int main(void) { return 0; }
