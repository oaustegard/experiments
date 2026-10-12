// EXPECT-EXIT: 0
// EXPECT-STDOUT: 40 49 0f db
// EXPECT-STDOUT: 78 56 34 12
// EXPECT-STDOUT: -2 1 -7 1 0x12345678 0xff00
// scalar_storage_order("big-endian"): scalar members (float, int, short, a
// short array) are stored big-endian; reads, writes, compound assignment and
// static initializers all convert the bytes.
#include <stdio.h>
struct __attribute__((scalar_storage_order("big-endian"))) R {
    float f;
    short s[2];
};
struct __attribute__((scalar_storage_order("little-endian"))) L { int i; };
struct __attribute__((scalar_storage_order("big-endian"))) G { int i; short h; };
struct R gr = { 3.1415927f, { 258, -7 } };
struct G gg = { 0x12345678, -2 };
static void dump(const void *p, int n) {
    const unsigned char *b = p;
    for (int i = 0; i < n; i++) printf("%s%02x", i ? " " : "", b[i]);
    printf("\n");
}
int main(void) {
    struct R r = { 3.1415927f, { 258, -7 } };
    struct L l = { 0x12345678 };
    struct G g = gg;
    g.i += 0x1000;
    g.h++;
    dump(&r, 4);
    dump(&l.i, 4);
    printf("%d %d %d %d 0x%x 0x%x\n", gg.h, gr.s[0] == 258, gr.s[1], (int)(r.f > 3.14f),
           (unsigned)gg.i, (unsigned short)g.h - 0xff);
    return 0;
}
