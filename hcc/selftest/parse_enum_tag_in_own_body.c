// EXPECT-EXIT: 0
// The tag of `enum A { ... }` is in scope inside its own body, so the
// `enum A *` in the _Generic refers to the inner incomplete type, not the
// outer global enum A.
enum A { B = 7 } y;
void g(void) {
    enum A { B = _Generic(&y, enum A *: 1, default: 7) };
    _Static_assert(7 == B, "");
}
int main(void) { return 0; }
