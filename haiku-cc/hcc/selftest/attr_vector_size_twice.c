// EXPECT-ERROR: [parse] invalid vector type
int __attribute__((vector_size(8), vector_size(8))) v;
