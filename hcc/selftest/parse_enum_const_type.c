// EXPECT-EXIT: 0
// Enumerator constants take C23 types: int if it fits, else the expression's
// type, and implicit overflow widens to the next signed/unsigned type.
// Once the enum has a non-int underlying type, its constants after the body have the enum type.
enum e { a = 0x7fffffff, b, c = 0xffffffffu, d };
_Static_assert(((typeof(b))-1) > 0, "b has the unsigned enum type");
_Static_assert(((typeof(c))-1) > 0, "c is unsigned int");
_Static_assert(((typeof(d))-1) > 0, "d is unsigned long long-like");
_Static_assert(sizeof(typeof(a)) == sizeof(enum e), "a has the enum type");
int main(void) { return 0; }
