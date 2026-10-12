// EXPECT-EXIT: 0
// GCC extension: reads of const static objects at constant offsets fold in static
// initializers (array element, struct member sub-array element).
static const char a[] = "0123456789";
struct S { const char c[4]; int n; };
static const struct S e[] = { { "abcd", 1 }, { "wxyz", 2 } };
static const char b = a[3];
static const char f = e[1].c[2];
int main(void) {
    return (b == '3' && f == 'y') ? 0 : 1;
}
