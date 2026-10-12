// EXPECT-EXIT: 0
// A member whose offset lies beyond imm32 (packed struct after a >2 GiB
// array) must be added to %rax through a 64-bit register.
struct __attribute__((packed)) A { char c; short a[__INT_MAX__]; };
struct S { struct A p; short tail; };
static struct S q;
int main(void) {
    q.tail = 7;
    return q.tail - 7;
}
