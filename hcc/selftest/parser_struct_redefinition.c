// EXPECT-ERROR: [parse]
// A second body for the same struct tag in one scope is an error.
struct r { int a; };
struct r { int b; };
int main(void) { return 0; }
