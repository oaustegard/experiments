// EXPECT-ERROR: [parse]
/* vector_size byte count above the signed size maximum must be rejected */
typedef char V __attribute__((vector_size(1ULL << 63)));
int main(void) { return 0; }
