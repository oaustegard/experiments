// EXPECT-EXIT: 0
// __imag__ of a real operand is 0, but its operand is still evaluated (PR 33192).
int main(void) {
    int i = 1;
    int j = __imag__ ++i;
    if (i != 2 || j != 0) return 1;
    double d = 2.5;
    if (__real__ d != 2.5) return 2;
    return 0;
}
