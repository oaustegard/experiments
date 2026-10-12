// EXPECT-EXIT: 0
// Extended asm: an empty constraint string ("") imposes no restriction and
// behaves like "g": the operand is placed in a general register.
int main(void) {
    int a = 5;
    asm volatile("" : : ""(a));
    asm volatile("" : "+&r"(a) : ""(a));
    return a == 5 ? 0 : 1;
}
