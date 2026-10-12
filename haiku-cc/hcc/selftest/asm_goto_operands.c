// EXPECT-EXIT: 0
// Extended asm: "+" outputs get an implicit input after the explicit inputs
// (so %N numbering skips past it), asm goto labels are operands after those,
// and a jump to a label goes through a stub that pops the output-address slot.
int f(int x) {
    int i = x;
    asm goto("jmp %l[hit]" : "+r"(i) ::: hit);
    return 1000;
hit:
    return i + 10;
}
int h(int x) {
    int i = x, j = 1;
    asm goto("# %0 %1 %2" : "+r"(i), "+r"(j) ::: done);
    i += 100;
done:
    return i + j;
}
int g(void) {
    int a = 5;
    asm("movq $-1, %q1" : "+a"(a));
    return a == -1 ? 0 : 1;
}
int main(void) {
    if (f(1) != 11) return 1;
    if (h(1) != 102) return 2;
    return g();
}
