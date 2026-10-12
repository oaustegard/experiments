// EXPECT-EXIT: 0
typedef int V4 __attribute__((vector_size(16)));
typedef __int128 S128 __attribute__((vector_size(16)));
int main(void) {
    V4 a = {1, -2, 3, 4}, b = {1, 2, 2, 9};
    V4 r = a < b;                          /* signed lanes: -1 true, 0 false */
    if (r[0] != 0 || r[1] != -1 || r[2] != 0 || r[3] != -1) return 1;
    V4 q = a == b;
    if (q[0] != -1 || q[1] != 0) return 2;
    V4 n = a != 4;                         /* scalar broadcast */
    if (n[3] != 0 || n[0] != -1) return 3;
    S128 x = {-1}, y = {1};
    S128 z = x < y;                        /* 128-bit signed compare */
    if (z[0] == 0) return 4;
    S128 w = x > y;
    if (w[0] != 0) return 5;
    return 0;
}
