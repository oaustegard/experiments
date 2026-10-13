// EXPECT-ERROR: read-only location
// A struct with a const member is not assignable, even through a non-const
// (volatile) variant declared before the struct was completed (C11 6.3.2.1p1).
struct foo s;
volatile struct foo t;
struct foo { const int z; };
int main(void) {
    t = s;
    return 0;
}
