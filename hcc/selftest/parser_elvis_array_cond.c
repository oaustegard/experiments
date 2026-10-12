// EXPECT-EXIT: 0
// GNU `a ?: b` where `a` is an array: the hidden temporary takes the decayed
// (pointer) type, so the condition is not an assignment to an array.
struct s { char c[1]; };
struct s a;
int main(void) {
    char *p = (a.c ? : 0);
    int x[2] = {0, 0};
    int *q = (p ? 0 : x);
    return (p != 0 && q == 0) ? 0 : 1;
}
