// EXPECT-ERROR: conflicting types for 'a'
// File-scope redeclarations must have compatible types (C11 6.7p4).
int a;
int a[] = {2};
