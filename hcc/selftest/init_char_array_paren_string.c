// EXPECT-EXIT: 0
// A parenthesized string literal inside braces initializes a char array.
char a[] = { ("foo") };
int main(void) { return a[0] == 'f' && a[2] == 'o' && a[3] == 0 ? 0 : 1; }
