// EXPECT-ERROR: [sema]
// A struct/union cannot be assigned from a scalar (here a function designator).
union u { int i; int (*fp)(int); };
extern int f1(int k);
void test(void) {
    union u a;
    a = f1;
}
