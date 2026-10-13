// EXPECT-EXIT: 0
// The "b" constraint pins an operand to rbx (callee-saved, preserved around the asm).
// Mbed TLS's x86-64 MULADDC reads %%rbx directly; without this every bignum op was wrong.
#include <stdint.h>
int main(void) {
    uint64_t b = 7, out;
    __asm__("movq %%rbx, %0" : "=r"(out) : "b"(b));
    uint64_t c = 5, s[2] = {3, 0}, d[2] = {0, 0};
    uint64_t *sp = s, *dp = d;
    __asm__("xorq %%r8, %%r8\n" "movq (%%rsi), %%rax\n" "mulq %%rbx\n" "addq $8, %%rsi\n"
            "addq %%rcx, %%rax\n" "movq %%r8, %%rcx\n" "adcq $0, %%rdx\n"
            "addq %%rax, (%%rdi)\n" "adcq %%rdx, %%rcx\n" "addq $8, %%rdi\n"
            : "+c"(c), "+D"(dp), "+S"(sp), "+m"(*(uint64_t (*)[2])d)
            : "b"(b), "m"(*(const uint64_t (*)[2])s) : "rax", "rdx", "r8");
    return (out == 7 && d[0] == 26 && c == 0) ? 0 : 1;
}
