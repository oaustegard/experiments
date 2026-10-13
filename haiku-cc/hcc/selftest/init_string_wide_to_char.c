// EXPECT-ERROR: [parse]
// A wide/UTF-16 string literal may not initialize a plain char array (C11 6.7.9p14).
const char s[] = u"ab";
