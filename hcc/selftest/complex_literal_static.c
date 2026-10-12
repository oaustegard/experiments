// EXPECT-EXIT: 0
// The imaginary part of an imaginary constant is discarded by a real conversion.
double e = 3.1i;
double d = 1.d;
_Complex float f = 1.0fi + 2.0f;
_Complex double z = 1.0 + 2.0i;
int main(void) {
    return (d == 1.0 && e == 0.0 && __imag__ z == 2.0 && __real__ f == 2.0f && __imag__ f == 1.0f) ? 0 : 1;
}
