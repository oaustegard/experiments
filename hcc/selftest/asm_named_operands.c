// EXPECT-EXIT: 0
// Extended asm: symbolic operand names ("%[x]" in the template and "[x]"
// as a matching input constraint) refer to the operand with that name.
int main(void) {
    int in = 5, out = 0;
    asm("movl %[src], %[dst]"
        : [dst] "=r" (out)
        : [src] "r" (in));
    asm("addl $1, %[v]"
        : [v] "+r" (out));
    asm("" : [o] "=r" (out) : "[o]" (out));
    return out == 6 ? 0 : 1;
}
