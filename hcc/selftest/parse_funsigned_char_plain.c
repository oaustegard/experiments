// EXPECT-EXIT: 0
// EXPECT-FLAGS: -std=c23 -funsigned-char
// With -funsigned-char plain char is unsigned: 0xff fits a constexpr char object,
// and a plain char object holds 255 at run time.
constexpr char v[] = "\xff";
int main(void) {
    char c = (char)0xff;
    return (c > 0) ? 0 : 1;
}
