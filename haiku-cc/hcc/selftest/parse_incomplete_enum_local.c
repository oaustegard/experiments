// EXPECT-ERROR: [parse]
// A block-scope object of a forward-declared enum type has no storage size.
enum E;
int main(void) { enum E e; return 0; }
