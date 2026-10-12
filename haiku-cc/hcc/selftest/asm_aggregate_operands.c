// EXPECT-EXIT: 10
// Extended asm: 1/2/4/8-byte struct operands with GPR constraints are passed
// as their bytes, for input, read-write and output operands.
struct S4 { int i; };
struct S1 { char c; };

int main(void) {
    struct S4 a = { 5 };
    struct S1 c = { 0 };
    asm("mov $3, %0" : "=r"(c));
    asm("" : "+r"(a));
    asm("" : : "r"(a));
    return a.i + c.c + 2;
}
