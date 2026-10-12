// EXPECT-ERROR: address of register variable
// A `register` local may not have its address taken.
void g(int *p);
int main(void) {
    register int x = 1;
    g(&x);
    return 0;
}
