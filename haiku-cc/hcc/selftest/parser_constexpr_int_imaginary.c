// EXPECT-ERROR: [parse]
// C23: an integer-typed constexpr object needs an integer constant initializer.
constexpr int i = 5i;
int main(void) { return i; }
