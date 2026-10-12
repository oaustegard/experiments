// EXPECT-EXIT: 0
// A const-qualified global struct's integer member is an integer constant.
struct s { int a; long b; };
const struct s cs = { 5, 7 };
int arr[cs.a - 3];
int main(void) {
    _Static_assert(cs.a == 5, "member folds");
    _Static_assert(cs.b == 7, "member folds");
    return arr[1] == 0 ? 0 : 1;
}
