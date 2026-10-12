// EXPECT-ERROR: [parse]
// Redeclaring a file-scope array with a different element type is a conflict
// (C11 6.2.7); also block-scope redeclaration with an incompatible type.
extern int a[];
extern unsigned a[3];
int main(void) { return 0; }
