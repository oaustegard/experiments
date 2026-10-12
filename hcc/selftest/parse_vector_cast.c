// EXPECT-EXIT: 0
typedef int V2SI __attribute__ ((vector_size (8)));
typedef short V2HI __attribute__ ((vector_size (4)));
typedef unsigned int V2USI __attribute__ ((vector_size (8)));
int main(void) {
    V2SI v = { 1, 2 };
    long long l = (long long) v;          /* vector -> scalar: same bytes */
    if ((int) l != 1) return 1;
    V2SI w = (V2SI) 0LL;                   /* scalar -> vector */
    if (w[0] != 0 || w[1] != 0) return 2;
    V2USI u = (V2USI) v;                   /* vector -> vector */
    if (u[1] != 2) return 3;
    int h = (int) (V2HI) 0;
    if (h != 0) return 4;
    return 0;
}
