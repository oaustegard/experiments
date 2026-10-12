// EXPECT-EXIT: 0
// A scalar initializer in braces may carry excess elements (gcc warns only).
int *p = {0, 0};
int x = {3, 4, 5};
int main(void) {
    int y = {7, 8};
    return (p == 0 && x == 3 && y == 7) ? 0 : 1;
}
