// EXPECT-EXIT: 0
// Builtin math functions called without a declaration get a double
// prototype: the result type is double (checked without emitting a call).
extern void abort(void);
int main(void) {
    unsigned long a = 1024;
    if (sizeof(__builtin_log2(a)) != sizeof(double)) abort();
    if (_Generic(__builtin_log2(a), double: 0, default: 1)) abort();
    if (_Generic(__builtin_log2f(a), float: 0, default: 1)) abort();
    return 0;
}
