// EXPECT-ERROR: [parse]
// An enum body may not redefine its own tag inside a sizeof in an enumerator.
enum A { N = sizeof(enum A { M = 1 }) };
