// EXPECT-EXIT: 0
typedef float v4 __attribute__((vector_size(sizeof(float)*4)));
int assrt[__alignof__(v4) == sizeof(float)*4 ? 1 : -1];
int main(void) { return 0; }
