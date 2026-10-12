// EXPECT-EXIT: 0
// C11 6.7.6.2: type qualifiers (including _Atomic) are allowed inside
// array declarator brackets, both bare and with a size.
void f(int [_Atomic]);
void g(int [_Atomic 2]);
int main(void) {
    int _Atomic x[3];
    return (int)sizeof x == 3 * (int)sizeof(int) ? 0 : 1;
}
