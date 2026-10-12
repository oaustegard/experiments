// EXPECT-ERROR: [parse]
// __atomic_load's first operand must be a non-void pointer (gcc PR 82112); an rvalue array does not decay.
struct S { int a[10]; } bar(void);
int b;
void f(void) {
  __atomic_load(bar().a, &b, 5);
}
