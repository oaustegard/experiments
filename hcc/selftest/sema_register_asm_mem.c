// EXPECT-ERROR: address of register variable
// A "m" asm operand naming a `register` local needs its address.
int main(void) {
    register int r = 1;
    __asm__("" : : "m"(r));
    return 0;
}
