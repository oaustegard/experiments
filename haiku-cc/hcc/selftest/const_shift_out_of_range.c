// EXPECT-EXIT: 0
// Shifts with a negative or over-wide constant count are undefined; they must
// not crash the constant folder (ValueError) and must not be folded.
int main(void) {
    int t = 1;
    unsigned c = 0;
    unsigned d = t ? 1 ^ c ^ 1 >> (-1) : 0;
    int e = t ? 1 << 40 : 0;
    return (int)(d & 0) + (e & 0);
}
