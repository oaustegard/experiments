// EXPECT-EXIT: 0
// Extended asm: a complex operand passed by address ("m") occupies one stack
// slot; popping it must not also pop a register pair (stack depth underflow).
#define H(x) asm("" : "=m"(x) : "m"(x))
int main(void) {
    _Complex double a = 1.0, b = 0.0;
    H(a);
    H(b);
    return 0;
}
