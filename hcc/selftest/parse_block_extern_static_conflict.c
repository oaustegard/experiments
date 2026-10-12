// EXPECT-ERROR: previously declared 'static'
static int v = 3;
void f(void) {
  int v = 4;
  {
    extern int v;
  }
}
