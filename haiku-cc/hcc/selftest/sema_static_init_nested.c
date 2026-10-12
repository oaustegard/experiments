// EXPECT-ERROR: initializer element is not constant
// A static initializer may not name a GNU nested function (it needs the frame).
struct S { void (*f)(int); };
void foo(void) {
  void bar(int v) { (void)v; }
  static struct S s = { bar };
  (void)s;
}
