// EXPECT-EXIT: 0
// __extension__ before a parenthesized expression is not a cast/type start.
int main(void) {
    int x = 5;
    int y = (__extension__ (x ? 5 : 2));
    __extension__ int z = (__extension__ (y));
    return z - 5;
}
