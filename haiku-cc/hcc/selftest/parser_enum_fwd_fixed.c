// EXPECT-ERROR: [parse]
// C23: an enum forward-declared without a fixed underlying type cannot later
// get one (gcc: 'enum' declared both with and without fixed underlying type).
enum e1;
enum e1 : int;
