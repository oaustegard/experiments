// EXPECT-EXIT: 0
// GNU: a difference of two labels plus an integer is a link-time constant, and a
// read of a string literal at a constant index folds to the character.
int main(void) {
    static long d = ((char *)&&l2 - (char *)&&l1) - 1;
    static int c = "abc"[1] + 2["abc"];
l1:
l2:
    if (d != -1) return 1;
    if (c != 98 + 99) return 2;
    return 0;
}
