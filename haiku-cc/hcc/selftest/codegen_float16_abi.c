// EXPECT-EXIT: 0
// _Float16 parameters, returns and variadic arguments use the half encoding.
extern void abort(void);
_Float16 twice(_Float16 x) { return x + x; }
_Float16 pass_va(int n, ...) { (void)n; return 0; }
double dbl(_Float16 x, _Float16 y) { return (double)(x * y); }
int main(void) {
    _Float16 h = 1.5f16;
    if (twice(h) != 3.0f16) abort();
    if (dbl(h, 2.0f16) != 3.0) abort();
    if (twice(twice(h)) != 6.0f16) abort();
    return 0;
}
