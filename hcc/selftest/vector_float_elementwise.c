// EXPECT-EXIT: 0
typedef float v4 __attribute__ ((vector_size (16)));
typedef double v2d __attribute__ ((vector_size (16)));
float a[8] = {1, 2, 3, 4, 5, 6, 7, 8};
float b[8] = {10, 20, 30, 40, 50, 60, 70, 80};
int main(void) {
    float d[4];
    v4 x = *(v4 *) a;
    v4 y = *(v4 *) b;
    v4 z = x + y;
    v4 m = x * y;
    *(v4 *) d = z;
    if (d[0] != 11 || d[3] != 44) return 1;
    *(v4 *) d = m;
    if (d[1] != 40) return 2;
    x -= y;
    *(v4 *) d = x;
    if (d[2] != -27) return 3;
    v2d p = { 1.5, 2.5 };
    v2d q = { 0.5, 0.5 };
    p /= q;
    if (p[0] != 3.0 || p[1] != 5.0) return 4;
    return 0;
}
