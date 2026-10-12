// EXPECT-EXIT: 0
int f(int n) {
    typedef int row[n + 1];
    typedef char grid[2][n];
    if (sizeof(row) != 4 * (n + 1)) return 1;
    if (sizeof(grid) != 2 * n) return 2;
    if (sizeof(row) * 2 != 8 * (n + 1)) return 3;
    return 0;
}
int main(void) {
    return f(5) + f(0) * 0;
}
