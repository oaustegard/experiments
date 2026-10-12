// EXPECT-EXIT: 0
// C23 `enum e : _Bool`: converting 0.5 to the enum yields 1 (as _Bool does).
enum eb : _Bool { BF, BT };
char d[(enum eb)0.5 == 1 ? 1 : -1];
char f[(enum eb)0.0 == 0 ? 1 : -1];
int main(void) { return 0; }
