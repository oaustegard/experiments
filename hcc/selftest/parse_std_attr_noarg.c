// EXPECT-ERROR: [parse]
// C23 standard attributes that take no arguments reject an argument list.
[[noreturn()]] void a(void);
