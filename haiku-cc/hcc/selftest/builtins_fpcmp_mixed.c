// EXPECT-EXIT: 1
// quiet comparison builtins accept an integer operand next to a floating one
int main(void) {
    double d = 1.0;
    float f = 2.0f;
    int a = __builtin_isunordered(d, 0);
    int b = __builtin_isunordered(0, f);
    int c = __builtin_isgreater(d, 0);
    return a + b + c;
}
