// EXPECT-ERROR: [parse]
// A tag may not be defined again inside its own body (struct and enum).
struct s { struct s { int i; } x; };
