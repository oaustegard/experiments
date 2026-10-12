// EXPECT-EXIT: 3
// Local VLA objects: storage is freed on block exit and on re-execution of
// the declaration via a backward goto (stack must not grow per iteration).
int main(void) {
    int n = 0, sum = 0;
lab:;
    {
        int x[n % 10 + 1];
        x[0] = 1;
        x[n % 10] = 2;
        sum = x[0] + x[n % 10] * 0 + (n % 10 == 0 ? 0 : 0);
        n++;
    }
    if (n < 100000) goto lab;
    {
        int v[3];
        v[0] = 0; v[1] = 1; v[2] = 2;
        sum = v[2];
    }
    return sum + 1;
}
