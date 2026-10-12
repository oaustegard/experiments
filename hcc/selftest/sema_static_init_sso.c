// EXPECT-ERROR: initializer element is not constant
// A scalar_storage_order struct cannot hold a relocated address constant.
int i;
struct __attribute__((scalar_storage_order("big-endian"))) R { int *p; };
struct R r = { &i };
