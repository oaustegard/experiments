// EXPECT-ERROR: unsupported builtin '__builtin_ia32_cvtpi2ps'
// An x86 intrinsic that is typed but not lowered is a clear codegen error when used.
typedef float v4sf __attribute__((vector_size(16)));
typedef int v2si __attribute__((vector_size(8)));
int main(void) {
    v4sf v = {1, 2, 3, 4};
    v2si m = {5, 6};
    v = __builtin_ia32_cvtpi2ps(v, m);
    return (int)v[0];
}
