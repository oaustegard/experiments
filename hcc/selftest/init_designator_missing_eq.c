// EXPECT-ERROR: [parse]
// `.member` without `=` is not the obsolete GNU form; only `[i] v` and `name: v` are.
struct s { int a; };
struct s s2 = { .a 1 };
