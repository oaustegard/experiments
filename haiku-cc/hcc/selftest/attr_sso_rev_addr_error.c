// EXPECT-ERROR: [parse]
// Taking the address of a scalar member of a scalar_storage_order("big-endian")
// struct is rejected (gcc does the same); it would otherwise yield a pointer
// to bytes that are not in native order.
struct __attribute__((scalar_storage_order("big-endian"))) R { int i; };
int main(void) {
    struct R r = { 1 };
    int *p = &r.i;
    return *p;
}
