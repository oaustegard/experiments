// EXPECT-EXIT: 3
// attribute between '(' and '*' in an abstract declarator (cast to fn pointer)
#define ATTR __attribute__((__noinline__))
int f(void) { return 3; }
int main(void) {
  void *fp = (void *)f;
  return ((int(ATTR *)(void))fp)();
}
