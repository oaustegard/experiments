// EXPECT-ERROR: conflicting types for 'a'
// File-scope redeclarations must have compatible types (array vs scalar).
extern int a;
int a[] = { 0 };
