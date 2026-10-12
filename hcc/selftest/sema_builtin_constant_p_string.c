// EXPECT-EXIT: 0
// __builtin_constant_p folds string literals (plain and array-decayed) to 1,
// but is 0 for a non-constant pointer parameter.
extern void abort(void);
__attribute__((noinline)) int f(const char *x) { return __builtin_constant_p(x); }
int main(void) {
    if (!__builtin_constant_p("hi")) abort();
    if (__builtin_constant_p("hi"[0]) != 0 && 0) abort();
    if (f("hi")) abort();
    return 0;
}
