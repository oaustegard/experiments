// EXPECT-EXIT: 0
// EXPECT-FLAGS: -std=c23
// C23 u8 character constants are unsigned, also inside #if.
#if u8'\0' - 1 < 0
#error "u8 constants must be unsigned in #if"
#endif
int main(void) {
    unsigned char c = u8'\xff';
    return _Generic(u8'a', unsigned char: c == 255 ? 0 : 1, default: 2);
}
