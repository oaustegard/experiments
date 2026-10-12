// EXPECT-EXIT: 10
// A statement expression whose block declares a VLA ends with a VLAExit item;
// its value is still the last expression statement (typed int, not void).
int main(void) {
    int n = 0;
    int r = sizeof *({ n = 10; int x[n]; &x; });
    if (r != (int)sizeof(int[10])) return 1;
    return (int)(*({ int m = 10; int y[m]; y[0] = 0; &y; }))[0] + 10;
}
