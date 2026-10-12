// EXPECT-EXIT: 0
// Extended asm: x87 ("t"/"u") operands may be float or double; the value is
// converted to long double on input and stored back with the C type's width.
int main(void) {
    double d = 4.0;
    double r;
    __asm__ volatile("" : "=t"(r) : "0"(d));
    if (r != 4.0) return 1;
    float f = 2.5f;
    float q;
    __asm__ volatile("" : "=t"(q) : "0"(f));
    if (q != 2.5f) return 2;
    return 0;
}
