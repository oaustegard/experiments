// EXPECT-ERROR: [parse]
// A member name may not repeat, including one reached through an anonymous union member.
struct S { int a; union { int a; }; };
