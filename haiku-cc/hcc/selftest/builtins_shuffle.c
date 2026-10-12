// EXPECT-EXIT: 0
// __builtin_shuffle: one- and two-operand forms, indices taken modulo N / 2N.
extern void abort(void);
typedef unsigned char V __attribute__((vector_size(4)));
typedef int IV __attribute__((vector_size(8)));
int main(void) {
    V a = {1, 2, 3, 4}, b = {5, 6, 7, 8};
    V m = {3, 4, 0, 7};
    V c = __builtin_shuffle(a, m);
    V d = __builtin_shuffle(a, b, m);
    if (c[0] != 4 || c[1] != 1 || c[2] != 1 || c[3] != 4) abort();
    if (d[0] != 4 || d[1] != 5 || d[2] != 1 || d[3] != 8) abort();
    IV x = {10, 20}, y = {30, 40};
    IV z = __builtin_shuffle(x, y, (IV){3, 1});
    if (z[0] != 40 || z[1] != 20) abort();
    return 0;
}
