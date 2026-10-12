// EXPECT-ERROR: [parse]
// Array sizes must have integer type (gcc PR 93573): a floating constant is rejected.
int f[100.0];
