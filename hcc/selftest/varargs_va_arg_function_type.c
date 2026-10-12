// EXPECT-ERROR: [parse]
// va_arg with a function type as its second argument is invalid.
int f(int n, ...) {
    __builtin_va_list ap;
    __builtin_va_start(ap, n);
    int r = __builtin_va_arg(ap, int (void))();
    return r;
}
