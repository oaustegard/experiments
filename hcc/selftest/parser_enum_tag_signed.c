// EXPECT-EXIT: 0
// A tagged enum with a negative enumerator is signed (gcc), and one with a value
// beyond 32 bits has an 8-byte underlying type (GNU extension, enum1.c).
enum e { a = -5, b = 3 };
enum wide { w = 0xffffffffffffffffULL, w0 = 0 };
int main(void) {
    enum e x = a;
    if (!(x < 0)) return 1;
    if (x > 0) return 2;
    if (sizeof(enum e) != 4) return 3;
    if (sizeof(enum wide) != sizeof(unsigned long long)) return 4;
    return 0;
}
