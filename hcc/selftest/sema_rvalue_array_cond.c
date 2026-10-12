// EXPECT-ERROR: [parse] used array that cannot be converted to pointer where scalar is required
// An array member of a function's struct result is an rvalue and cannot decay in a condition.
struct s { char c[1]; };
struct s f(void);
int main(void) { return f().c ? 1 : 0; }
