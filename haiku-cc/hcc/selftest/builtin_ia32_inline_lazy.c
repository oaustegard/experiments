// EXPECT-EXIT: 14
// gnu_inline `extern inline` bodies are emitted only when referenced: the unused
// one below uses an x86 builtin that codegen does not lower and must not be emitted.
typedef float v4sf __attribute__((vector_size(16)));
typedef int v2si __attribute__((vector_size(8)));
extern __inline __attribute__((__gnu_inline__, __always_inline__)) int twice(int x) { return 2 * x; }
extern __inline __attribute__((__gnu_inline__)) int unused_sse(v4sf v, v2si m) {
    __builtin_ia32_cvtpi2ps(v, m);
    return 0;
}
int main(void) { return twice(7); }
