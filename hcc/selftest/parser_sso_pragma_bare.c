// EXPECT-EXIT: 0
// A bare or unknown '#pragma scalar_storage_order' only warns (GCC behaviour).
#pragma scalar_storage_order
#pragma scalar_storage_order bogus
#pragma scalar_storage_order big-endian
struct s { int i; };
int main(void) { return 0; }
