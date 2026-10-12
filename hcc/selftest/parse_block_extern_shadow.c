// EXPECT-ERROR: [parse]
int a[1] = { 0 };
void f(void) {
  int a;
  {
    extern int a[];
    sizeof (a);
  }
}
