// EXPECT-EXIT: 0
// A weak function declaration with no definition in this TU must link:
// its address is an undefined weak reference that resolves to 0.
extern int maybe_missing(void) __attribute__((weak));
int main(void) {
    int (*f)(void) = maybe_missing;
    return f != 0;
}
