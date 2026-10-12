// EXPECT-ERROR: [parse]
// A function definition may not take a parameter of a forward-declared enum type.
enum E;
void foo (enum E e) { }
