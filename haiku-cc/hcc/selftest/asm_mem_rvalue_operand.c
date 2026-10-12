// EXPECT-EXIT: 12
// A "m" input that is not an lvalue (statement expression) is spilled to a temp.
int g = 5;
int main(void) {
  int r;
  __asm__ ("movl %1, %0" : "=r"(r) : "m"(({unsigned v = 7; v + g;})));
  return r;
}
