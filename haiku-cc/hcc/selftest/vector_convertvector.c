// EXPECT-EXIT: 0
// __builtin_convertvector converts lanes element-wise; float vector compares
// yield -1/0 lanes.
typedef int V2I __attribute__((vector_size(8)));
typedef double V2D __attribute__((vector_size(16)));
typedef long long V2L __attribute__((vector_size(16)));
int main(void) {
    V2I a = {1, -2};
    V2D d = __builtin_convertvector(a, V2D);
    if (d[0] != 1.0 || d[1] != -2.0) return 1;
    V2D x = {1.0, 3.0};
    V2D y = {2.0, 2.0};
    V2L m = x < y;
    if (m[0] != -1 || m[1] != 0) return 2;
    V2L g = x > y;
    if (g[0] != 0 || g[1] != -1) return 3;
    return 0;
}
