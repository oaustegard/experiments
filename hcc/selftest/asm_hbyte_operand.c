// EXPECT-EXIT: 0
// Extended asm: %h selects the high-byte register of a "Q" (a/b/c/d) operand.
int main(void) {
    unsigned short x = 0x1200;
    unsigned char hi;
    __asm__("movb %h1, %0" : "=r"(hi) : "Q"(x));
    return hi != 0x12;
}
