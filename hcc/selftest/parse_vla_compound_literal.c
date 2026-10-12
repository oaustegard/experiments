// EXPECT-ERROR: [parse]
// A compound literal of variable length type is rejected, even when the
// length is a block-scope const int (which is not an integer constant expression).
int main(void) {
    const int i = 1;
    char *p = (char [i]){ "" };
    return p != 0;
}
