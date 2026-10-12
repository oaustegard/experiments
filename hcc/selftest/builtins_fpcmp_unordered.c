// EXPECT-EXIT: 0
// C99 quiet comparison builtins: NaN operands make every relation false
// except isunordered (true) and islessgreater (false).
int main(void) {
    double n = __builtin_nan("");
    double one = 1.0, two = 2.0;
    long double lone = 1.0L, ltwo = 2.0L;
    if (!__builtin_isunordered(n, one)) return 1;
    if (__builtin_isunordered(one, two)) return 2;
    if (__builtin_isgreater(n, one) || __builtin_isless(one, n)) return 3;
    if (__builtin_islessgreater(n, one)) return 4;
    if (!__builtin_isgreater(two, one) || __builtin_isgreater(one, two)) return 5;
    if (!__builtin_isless(one, two) || __builtin_isless(two, one)) return 6;
    if (!__builtin_isgreaterequal(one, one) || !__builtin_islessequal(one, one)) return 7;
    if (!__builtin_islessgreater(one, two)) return 8;
    if (!__builtin_isgreater(ltwo, lone) || __builtin_isgreater(lone, ltwo)) return 9;
    return 0;
}
