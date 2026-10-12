// EXPECT-EXIT: 0
// __builtin_acosh/__builtin_ldexp etc. have their libm return types, and
// two-operand functions with int operands are not given a one-operand prototype.
extern __typeof (__builtin_acosh (0)) t1;
extern double t1;
extern __typeof (__builtin_cbrtf (0)) t2;
extern float t2;
int main(void) {
    double d = __builtin_ldexp(1.0, 3);
    return d == 8.0 ? 0 : 1;
}
