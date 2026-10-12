// EXPECT-ERROR: incompatible types
// A struct value returned from a function with scalar return type is an error.
struct s { int a; };
int f(struct s s) { return s; }
int main(void) { return 0; }
