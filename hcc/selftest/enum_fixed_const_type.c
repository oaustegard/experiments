// EXPECT-EXIT: 0
// C23: enumerators of a fixed-underlying-type enum have the enum type; a short-based
// enum promotes to int, so `typeof (+e)` is compatible with `int`.
enum e1 : short { e1a = 5, e1b = -5 };
extern enum e1 e1v;
extern typeof (e1a) e1v;
extern typeof (+e1a) i;
extern int i;
int main(void) { return e1a + e1b; }
