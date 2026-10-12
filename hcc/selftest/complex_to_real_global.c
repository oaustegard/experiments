// EXPECT-EXIT: 0
// C11 6.3.1.7: a complex constant initializing a real file-scope object keeps
// only its real part (imaginary constants are complex-typed).
double a = 0.5i;
double b = 2.0 + 3.0i;
int c = 7i;
int main(void) {
    return (a == 0.0 && b == 2.0 && c == 0) ? 0 : 1;
}
