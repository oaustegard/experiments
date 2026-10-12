// EXPECT-EXIT: 0
// A plain typedef redeclaration keeps the largest alignment seen so far.
typedef int T;
typedef int T __attribute__((aligned (64)));
typedef int T;
_Static_assert (_Alignof (T) == 64, "sticky alignment");
int main(void) { return 0; }
