// EXPECT-ERROR: non-floating-point
// quiet comparison builtins require real floating operands
int main(void) {
    int i = 1;
    return __builtin_isgreater(i, i);
}
