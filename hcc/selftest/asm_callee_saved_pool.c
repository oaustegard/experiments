// EXPECT-EXIT: 0
// Extended asm: more register operands than the caller-saved pool holds spill
// into callee-saved registers, which must be preserved across the asm.
int a, b, c, d, e, f, g, h, i, j, k, l;
int main(void) {
    int keep = 7;
    asm volatile(""
        : [a] "+r" (a), [b] "+r" (b), [c] "+r" (c), [d] "+r" (d),
          [e] "+r" (e), [f] "+r" (f), [g] "+r" (g), [h] "+r" (h),
          [i] "+r" (i), [j] "+r" (j), [k] "+r" (k), [l] "+r" (l));
    asm volatile("" : : : "rbx", "r12");
    return keep == 7 ? 0 : 1;
}
