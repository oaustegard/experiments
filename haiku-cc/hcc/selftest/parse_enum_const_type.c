// EXPECT-EXIT: 0
// Enumerator constants take C23 types: int if it fits, else the expression's
// type, and implicit overflow widens to the next signed/unsigned type.
enum e { a = 0x7fffffff, b, c = 0xffffffffu, d };
_Static_assert(((typeof(b))-1) < 0, "b is signed");
_Static_assert(((typeof(c))-1) > 0, "c is unsigned int");
_Static_assert(((typeof(d))-1) > 0, "d is unsigned long long-like");
_Static_assert(sizeof(typeof(a)) == 4, "a is int");
int main(void) { return 0; }
