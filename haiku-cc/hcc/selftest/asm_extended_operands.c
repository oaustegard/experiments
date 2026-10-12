// EXPECT-EXIT: 0
// Extended asm: "+r" and "=r" outputs are stored back; "0" ties an input to an output.
extern void abort(void);
__attribute__((noinline)) void g(int x) { asm volatile("" : "+r"(x)); }
int main(void) {
    int l;
    asm("" : "=r"(l) : "0"(0));
    if (l != 0) abort();
    int x = 5;
    asm("addl %1, %0" : "+r"(x) : "r"(4));
    if (x != 9) abort();
    int z;
    asm("movl %1, %0" : "=&r"(z) : "0"(9));
    if (z != 9) abort();
    g(z);
    return 0;
}
