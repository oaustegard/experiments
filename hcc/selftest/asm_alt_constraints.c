// EXPECT-EXIT: 0
// Extended asm: constraint alternatives ("1,p", "=m,m") use the first
// alternative; X and p operands are accepted as general registers.
int main(void) {
    int b = 3;
    int *t = &b;
    asm volatile("" : "=m,m"(b), "=r,r"(b) : "1,p"(b));
    asm("" : "=X"(*t));
    asm("" : : "X"(b));
    return 0;
}
