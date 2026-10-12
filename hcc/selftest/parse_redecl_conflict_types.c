// EXPECT-ERROR: [parse]
// Same-scope redeclaration of an object with an incompatible type is an error.
double x;
int x;
int main(void) { return 0; }
