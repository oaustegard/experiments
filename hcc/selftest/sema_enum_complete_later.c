// EXPECT-EXIT: 0
// An object of enum type declared before the enumerator body is usable after it.
enum e;
extern enum e ve;
enum e { A, B };
enum e ve = B;
int main(void) { int x = ve; return x - 1; }
