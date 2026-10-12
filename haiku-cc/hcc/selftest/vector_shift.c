// EXPECT-EXIT: 0
typedef short V4HI __attribute__ ((vector_size (8)));
typedef unsigned short V4UHI __attribute__ ((vector_size (8)));
int main(void) {
    V4HI a = {1, 2, 3, 4}, b;
    V4UHI u = {0x8000, 4, 8, 1}, w;
    b = 2 << a;
    if (b[0] != 4 || b[3] != 32) return 1;
    b = a >> 1;
    if (b[1] != 1 || b[3] != 2) return 2;
    b = -a >> 1;
    if (b[0] != -1) return 3;
    w = u >> 1;
    if (w[0] != 0x4000) return 4;
    return 0;
}
