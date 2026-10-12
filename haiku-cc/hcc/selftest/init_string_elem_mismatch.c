// EXPECT-ERROR: [parse]
// A UTF-16 string literal may not initialize an array of plain short.
const short s[] = u"ab";
