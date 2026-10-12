// EXPECT-ERROR: not representable in type of object
// C23 constexpr: a plain string's \xff is a char value (-1), not representable in unsigned char.
// EXPECT-FLAGS: -std=c23
constexpr unsigned char s[] = "\x00\xff";

int main(void) { return s[0]; }
