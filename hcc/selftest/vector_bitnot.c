// EXPECT-EXIT: 0
typedef int v4si __attribute__((vector_size(16)));
typedef char v3qi __attribute__((vector_size(3)));
v4si g;
int main(void) {
    v4si a = {1, 2, 3, 4};
    g = ~a;
    if (g[0] != -2 || g[3] != -5) return 1;
    v4si b = ~g;
    if (b[1] != 2) return 2;
    return 0;
}
