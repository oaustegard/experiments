// EXPECT-EXIT: 0
// Any 1-byte string literal (u8, plain) may initialize a char-sized array.
const unsigned char a[] = u8"ab";
const signed char b[] = "cd";
int main(void) { return a[0] - 'a' + b[1] - 'd'; }
