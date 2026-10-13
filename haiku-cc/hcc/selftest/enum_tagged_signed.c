// EXPECT-EXIT: 0
// A tagged enum with a negative enumerator is signed int-compatible, so it
// matches an int association in _Generic.
enum E { E1 = -1, E2 = 0 };
int main(void) {
    int r = _Generic((int){0}, enum E: 1, default: 0);
    return r == 1 ? 0 : 1;
}
