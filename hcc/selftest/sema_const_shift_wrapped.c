// EXPECT-EXIT: 0
// A shift count computed as a size_t difference that wraps negative must not
// crash constant folding (it used to raise MemoryError); the expression is
// evaluated at runtime instead.
int main(void) {
    unsigned char c = 0;
    unsigned long n = sizeof(c) * 8 - 1 - 24;
    unsigned char v = (unsigned char)((1 << 7) + ((unsigned char)1 << (sizeof(c) * 8 - 1 - 24)));
    c = v;
    return c == 128 || c == 0 ? 0 : 1;
}
