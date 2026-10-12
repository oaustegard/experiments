// EXPECT-ERROR: second argument to 'va_arg' is of incomplete type
// va_arg's type argument must be a complete object type (C11 7.16.1.1p2); void is not.
int foo(__builtin_va_list ap) { return __builtin_va_arg(ap, void); }
int main(void) { return 0; }
