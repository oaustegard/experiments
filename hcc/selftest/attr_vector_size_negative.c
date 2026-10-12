// EXPECT-ERROR: is negative
int v __attribute__((vector_size(-1)));
int main(void) { return 0; }
