// EXPECT-EXIT: 0
int count = 0;
static int dummy;
static int *bar(void) { ++count; return &dummy; }
__attribute__((aligned(256))) void func(void);
void func(void) {}
int main(void) {
    asm("" : "+r"(*bar()));
    if (count != 1) return 1;
    if (__alignof__(func) != 256) return 2;
    return 0;
}
