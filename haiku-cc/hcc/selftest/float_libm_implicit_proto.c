// EXPECT-EXIT: 0
// Implicit calls of libm builtins take their operands at the function's own
// float type: the float argument is not promoted to double.
int main(void) {
    float f = -2.0f;
    float g = __builtin_copysignf(1.0, f);
    float h = __builtin_copysignf(3.0f, 1.0f);
    if (g != -1.0f) return 1;
    if (h != 3.0f) return 2;
    if (__builtin_copysign(1.0, (double)f) != -1.0) return 3;
    return 0;
}
