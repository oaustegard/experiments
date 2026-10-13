// EXPECT-ERROR: [parse] too few arguments to function
// __builtin_va_start with no arguments is a diagnosed error, not a codegen crash.
void f(char *fmt, ...) {
    __builtin_va_start();
}
