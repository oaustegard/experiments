// EXPECT-EXIT: 0
// typeof_unqual drops top-level const/volatile of its operand's type.
const volatile int ci = 5;
__typeof_unqual__(ci) x;
int main(void) {
    __typeof_unqual__(ci) y = 3;
    typeof(ci) z = 4;
    x = 1;
    if (x + y + z != 8) return 1;
    return 0;
}
