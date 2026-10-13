// EXPECT-EXIT: 0
// `int *_Atomic` and `_Atomic (int *)` are the same type, so the redeclaration is fine.
extern int *_Atomic b;
extern _Atomic (int *) b;
int main(void) { return 0; }
