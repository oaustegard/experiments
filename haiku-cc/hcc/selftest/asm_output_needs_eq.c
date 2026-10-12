// EXPECT-ERROR: [parse] output operand constraint lacks '='
// Output operands of extended asm must have '=' or '+' in their constraint.
int main(void) {
    int x = 0;
    __asm__ ("" : "r" (x));
    return x;
}
