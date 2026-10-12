// EXPECT-ERROR: [parse]
// An implicit enumerator past the largest __int128 value overflows.
enum e { a = (__int128)(((unsigned __int128)-1) >> 1), b };
