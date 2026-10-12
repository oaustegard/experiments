// EXPECT-EXIT: 0
// GCC-extension folds in static initializers: pointer differences of one base,
// reads of const integer globals, and &object compared with 0.
static const int k = 3;
static int obj[4];
static int *p0 = obj + 2;
long a1 = (long)(obj + 2) - (long)obj;
long a2 = (char *)&obj[3] - (char *)obj;
long a3 = k * 2;
int a4 = (&obj[0] >= 0) + (&obj[0] == 0) * 10;
int main(void) {
    int i[4];
    int *p = i + 2;
    static long b = p - (p - 1);
    if (a1 != 8 || a2 != 12 || a3 != 6 || a4 != 1) return 1;
    if (b != 1 || (p - 2) - (p - 1) != -1) return 2;
    if (p0 != obj + 2) return 3;
    return 0;
}
