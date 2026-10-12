// EXPECT-ERROR: [parse]
// GCC: the mode attribute on a struct body is an error ("applied to inappropriate type").
typedef struct __attribute__((mode(SI))) { int a; } S;
