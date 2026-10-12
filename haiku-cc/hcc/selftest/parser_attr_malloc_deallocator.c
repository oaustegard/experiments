// EXPECT-ERROR: [parse]
int var;
void* alloc_var(int) __attribute__((malloc(var)));
