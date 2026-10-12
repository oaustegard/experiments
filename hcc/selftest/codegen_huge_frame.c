// EXPECT-EXIT: 0
// A frame larger than 2 GiB must assemble: the prologue subtracts the frame size
// through a register, and locals beyond disp32 get their address via a computed
// base.  f is compiled but never called (running it would overflow the stack).
int f(void) {
    char local[0x80000000UL];
    char *p = &local[0x7fffffff];
    char *q = local;
    return p - q == 0x7fffffff;
}
int main(void) {
    return 0;
}
