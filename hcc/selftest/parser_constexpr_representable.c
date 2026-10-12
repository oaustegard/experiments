// EXPECT-ERROR: not representable in type of object
// EXPECT-FLAGS: -std=c23
// C23: a constexpr object's initializer values must be representable in its type.
constexpr signed char v[] = { 1, 129 };

int main(void) { return v[0]; }
