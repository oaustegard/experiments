// EXPECT-ERROR: [parse]
// C23: attributes may not prefix a _Static_assert declaration.
[[]] _Static_assert(1, "x");
int main(void) { return 0; }
