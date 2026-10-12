// EXPECT-EXIT: 0
// EXPECT-STDOUT: 40 49 0f db c0 49 0f db
// EXPECT-STDOUT: 40 49 0f db c0 49 0f db
// scalar_storage_order("big-endian") on a _Complex float member: each part is
// stored big-endian; assigning a native complex value converts part by part.
struct __attribute__((scalar_storage_order("big-endian"))) R { _Complex float F; };
struct R g;
int main(void) {
    unsigned char *p = (unsigned char *)&g;
    int i;
    g.F = 3.1415927f - 3.1415927f * (__extension__ 1.0iF);
    for (i = 0; i < 8; i++) __builtin_printf("%02x%s", p[i], i == 7 ? "\n" : " ");
    struct R l;
    l.F = g.F;
    unsigned char *q = (unsigned char *)&l;
    for (i = 0; i < 8; i++) __builtin_printf("%02x%s", q[i], i == 7 ? "\n" : " ");
    return 0;
}
