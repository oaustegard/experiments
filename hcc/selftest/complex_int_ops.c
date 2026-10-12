// EXPECT-EXIT: 0
// GNU _Complex int: inline integer arithmetic (matches GCC's lowering).
_Complex int mul(_Complex int a, _Complex int b) { return a * b; }
int main(void) {
    _Complex int a = 7, b = 2;
    __imag__ a = 3;
    __imag__ b = -5;
    _Complex int c = mul(a, b);
    if (__real__ c != 29 || __imag__ c != -29) return 1;
    if (a == b || !(a != b)) return 2;
    if (__real__ (-a) != -7 || __imag__ (-a) != -3) return 3;
    return 0;
}
