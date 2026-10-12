// EXPECT-ERROR: [parse]
// EXPECT-FLAGS: -std=c23
// C23 constexpr objects: every initializer value must be representable in the
// element type (129 does not fit in signed char).
constexpr signed char foo[3] = { 1, 2, 129 };
