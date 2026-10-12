// EXPECT-EXIT: 0
// Statements after __builtin_unreachable() in the same block are dead and
// must not be emitted (a call to an undefined function there must not link).
void never_defined(void);
int main(void) {
    volatile int x = 1;
    if (x == 2) {
        __builtin_unreachable();
        never_defined();
    }
    return 0;
}
