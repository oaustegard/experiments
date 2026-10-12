// EXPECT-EXIT: 0
// Extended asm: array-typed "g" inputs decay to their address.
extern void abort(void);
int main(void) {
    int a[4] = {1, 2, 3, 4};
    unsigned long long q[2] = {5, 6};
    asm volatile("" : : "g"(a), "g"(q) : "memory");
    return 0;
}
