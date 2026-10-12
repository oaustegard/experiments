// EXPECT-EXIT: 0
// Objects larger than 2 GiB put member offsets beyond the imm32 range of
// `add $imm`; codegen must go through a 64-bit register. A static object
// keeps the run within memory.
struct __attribute__((packed)) A { char c; short a[__INT_MAX__]; };
struct S { struct A p; short tail; };
static struct S q;
int main(void) {
    q.tail = 7;
    return q.tail - 7;
}
