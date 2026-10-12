// EXPECT-EXIT: 0
// __builtin_copysign without a declaration returns double (not implicit int).
int main(void) {
    double r = __builtin_copysign(1.0, -0.0);
    if (r != -1.0) return 1;
    if (__builtin_copysign(2.0, 3.0) != 2.0) return 2;
    return 0;
}
