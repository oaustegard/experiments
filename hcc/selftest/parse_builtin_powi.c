// EXPECT-EXIT: 0
// __builtin_powi{f,,l} lower to libgcc helpers, not to a user function of the same name.
float powif(float x, int n) { return __builtin_powif(x, n); }
double powi(double x, int n) { return __builtin_powi(x, n); }
int main(void) {
    if (powif(2.0f, 3) != 8.0f) return 1;
    if (powi(2.0, 4) != 16.0) return 2;
    return 0;
}
