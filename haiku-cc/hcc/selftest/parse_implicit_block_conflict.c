// EXPECT-EXIT: 7
// A block-scope declaration that conflicts with an earlier implicit declaration
// is only a GCC warning; the program still compiles and links.
void f(void)
{
  g();
  {
    void g();
  }
}
int g(void) { return 7; }
int main(void) { f(); return g(); }
