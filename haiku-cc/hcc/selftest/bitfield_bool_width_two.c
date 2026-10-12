// EXPECT-ERROR: [parse]
// _Bool bit-fields are limited to one bit.
struct s { _Bool b : 2; };
