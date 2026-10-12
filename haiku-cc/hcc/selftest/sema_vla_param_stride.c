// EXPECT-EXIT: 0
// A parameter `char a[2][n]` is a pointer to a VLA: the row stride is the
// run-time value of n (copied on entry), not a compile-time constant.
int N = 3;
void set(char a[2][N]) { a[1][2] = 7; }
int get(int n, char a[][n], int i, int j) { return a[i][j]; }
int main(void) {
    char buf[6] = {0};
    set((char (*)[3])buf);
    if (buf[5] != 7) return 1;
    if (buf[2] != 0) return 2;
    N = 2;
    char b[2][2] = {{1, 2}, {3, 4}};
    if (get(2, (char (*)[2])b, 1, 0) != 3) return 3;
    return 0;
}
