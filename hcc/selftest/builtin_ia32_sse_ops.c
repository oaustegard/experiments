// EXPECT-EXIT: 0
// EXPECT-STDOUT: 1 32767 -3 4 5 6 7 8
// EXPECT-STDOUT: 4 -3 70000 1
// EXPECT-STDOUT: 70000 5
// EXPECT-STDOUT: 32767 -5 3 -32768
// x86 SSE/MMX intrinsics as GCC's headers call them: packs with saturation,
// pshufd with an immediate, vec_ext, and an MMX pack (__m64 sized vectors).
typedef int v4si __attribute__((vector_size(16)));
typedef short v8hi __attribute__((vector_size(16)));
typedef int v2si __attribute__((vector_size(8)));
typedef short v4hi __attribute__((vector_size(8)));
#include <stdio.h>
int main(void) {
    v4si a = {1, 70000, -3, 4};
    v4si b = {5, 6, 7, 8};
    v8hi h = __builtin_ia32_packssdw128(a, b);
    for (int i = 0; i < 8; i++) printf(i ? " %d" : "%d", h[i]);
    printf("\n");
    v4si s = __builtin_ia32_pshufd(a, 0x1b);
    printf("%d %d %d %d\n", s[0], s[1], s[2], s[3]);
    printf("%d %d\n", __builtin_ia32_vec_ext_v4si(a, 1), __builtin_ia32_vec_ext_v4si(b, 0));
    v2si x = {70000, -5}, y = {3, -40000};
    v4hi q = (v4hi)__builtin_ia32_packssdw(x, y);
    printf("%d %d %d %d\n", q[0], q[1], q[2], q[3]);
    return 0;
}
