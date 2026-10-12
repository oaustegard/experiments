// EXPECT-ERROR: [parse]
// A bit-field wider than its declared type is a hard error (gcc: "width of 'i' exceeds its type").
struct s { int i : 40; };
