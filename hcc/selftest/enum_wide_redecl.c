// EXPECT-EXIT: 0
// An enum with a 64-bit underlying type: its constants have the enum type and
// redeclarations of the object may name the constant's type.
enum e1 { e1a = -__LONG_LONG_MAX__ - 1, e1b = 0, e1c = __LONG_LONG_MAX__ };
extern enum e1 e1v;
extern typeof(e1a) e1v;
extern typeof(e1b) e1v;
_Static_assert(e1a < 0, "signed 64-bit enum");
_Static_assert(e1c > 0, "positive");
int main(void) { return 0; }
