// EXPECT-EXIT: 0
// A __builtin_ name used without a call (&__builtin_abs) names the library function.
extern void abort(void);
int main(void) {
    int (*f)(int) = __builtin_abs;
    if (f(-3) != 3) abort();
    return 0;
}
