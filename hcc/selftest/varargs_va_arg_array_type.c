// EXPECT-ERROR: [parse]
int f(int n, ...)
{
    __builtin_va_list ap;
    int *a = __builtin_va_arg(ap, int[3]);
    return a != 0;
}
