// EXPECT-OK
// A block-scope prototype naming the old-style `()` function being defined
// is accepted inside its own body (GCC behaviour), even with more parameters.
void foo ();
void foo ()
{
  void foo (int);
}
int main(void) { return 0; }
