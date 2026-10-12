// EXPECT-ERROR: [parse]
// An array whose total size exceeds PTRDIFF_MAX is rejected (GCC: "exceeds maximum object size").
int main(void) { static char a[1 << 22][1 << 22][1 << 22]; return 0; }
