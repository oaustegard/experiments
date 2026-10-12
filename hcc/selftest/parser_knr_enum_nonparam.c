// EXPECT-ERROR: declared as a non-parameter
// An enumerator declared in an old-style parameter declaration list must not name a parameter.
int f(a)
  enum b { a };
{ return a; }
int main(void) { return 0; }
