// EXPECT-EXIT: 0
// Compound literals inside static-storage initializers become static objects;
// a scalar one folds to its value (GCC extension).
struct S { int a, b; };
struct S g = (struct S){ 4, 5 };
struct T { int i; };
struct T t = { .i = (int){ 9 } };
int main(void) {
    static struct S x = (struct S){ 1, 2 };
    static int *p = (int[]){ 6, 7 };
    if (x.a + x.b != 3 || p[1] != 7 || g.b != 5 || t.i != 9) return 1;
    return 0;
}
