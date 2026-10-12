// EXPECT-EXIT: 0
typedef short v8hi __attribute__((vector_size(16)));
typedef unsigned char v4qu __attribute__((vector_size(4)));
v8hi a = {150, 100, 150, 200, 0, 0, 0, 0};
v8hi b = {10, 13, 20, 30, 1, 1, 1, 1};
int main(void) {
    v8hi s = a + b, m = a * b, d = a / b, r = a % b, n = -a;
    if (s[0] != 160 || s[1] != 113) return 1;
    if (m[3] != 6000 || d[2] != 7 || r[2] != 10) return 2;
    if (n[0] != -150 || (a - b)[1] != 87) return 3;
    v4qu x = {200, 1, 2, 3}, y = {100, 1, 1, 1};
    v4qu z = x + y;
    if (z[0] != 44 || z[3] != 4) return 4;
    return 0;
}
