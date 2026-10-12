// EXPECT-ERROR: conflicting types for 'strcmp'
// A file-scope prototype must match the builtin strcmp signature.
int strcmp(void);
