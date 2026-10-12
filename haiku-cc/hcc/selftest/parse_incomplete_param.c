// EXPECT-ERROR: [parse]
// A function definition may not take a parameter of incomplete struct type.
struct A;
void foo (struct A a) { }
