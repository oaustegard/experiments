// EXPECT-ERROR: [parse]
/* a vector with more components than GCC allows (INT_MAX - 1) must be rejected */
typedef int V __attribute__((vector_size(1ULL << 34)));
int main(void) { return 0; }
