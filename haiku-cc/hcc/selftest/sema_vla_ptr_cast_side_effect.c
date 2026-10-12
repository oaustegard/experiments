// EXPECT-EXIT: 0
extern void abort(void);
int i;
void *p;
void f(void *x, int j) {
  p = (int (*)[++i])x;
  (void)(int (*)[++i]){p};
  if (i != j) abort();
}
int main(void) {
  int n = 3;
  int (*q)[n];
  int k = 0;
  q = (int (*)[++k])p;
  if (k != 1) abort();
  f(p, 2);
  return 0;
}
