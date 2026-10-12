// EXPECT-EXIT: 0
// An enum tag is visible inside its own enumerator list; a new block-scope tag hides the outer one.
enum A { B = 7 } y;
int main(void) {
    enum A { B = _Generic(&y, enum A *: 1, default: 7) };
    return B - 7;
}
