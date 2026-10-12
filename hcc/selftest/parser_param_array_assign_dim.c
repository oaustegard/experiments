// EXPECT-EXIT: 7
// Parameter array dimensions are assignment-expressions: "a[i = 0]" must parse
// and its side effect must run when the function is called.
int i = 5;
void fn1(int a[i = 0]) { (void)a; }
void fn2(int a[i += 2][3]) { (void)a; }
int main(void) {
    int a[10];
    fn1(a);
    if (i != 0) return 1;
    fn2((int (*)[3])a);
    if (i != 2) return 2;
    return 7;
}
