// EXPECT-ERROR: [parse] attribute 'scalar_storage_order' argument must be one of
// scalar_storage_order accepts only "big-endian" or "little-endian".
struct S { int i; };
typedef struct S __attribute__((scalar_storage_order("other"))) S3;
int main(void) { return 0; }
