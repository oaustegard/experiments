// EXPECT-EXIT: 7
// GNU `a ?: b` with a complex condition: the value is the complex condition
// itself (not its truth value), and the else arm is taken only when it is zero.
int main(void) {
    __complex__ int z = 0;
    __complex__ int v = (__complex__ int)(3 + 4i) ?: z;
    __complex__ int w = z ?: (__complex__ int)(3 + 4i);
    return __real__ v + __imag__ v - __real__ w - __imag__ w + 7 - 0;
}
